"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

# W&B is intentionally disabled in this training environment.
# The OpenVLA package metadata may list it as a dependency, but this script
# does not require it for fine-tuning.
wandb = None
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    checkpoint_steps: Optional[str] = None                           # Optional comma-separated optimizer
                                                                     # steps to save, e.g. "5000,7000".
                                                                     # When set, overrides periodic
                                                                     # saving through save_steps.
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    explicit_checkpoint_steps: Optional[set[int]] = None

    if cfg.checkpoint_steps is not None and cfg.checkpoint_steps.strip():
        try:
            explicit_checkpoint_steps = {
                int(item.strip())
                for item in cfg.checkpoint_steps.split(",")
                if item.strip()
            }
        except ValueError as exc:
            raise ValueError(
                "checkpoint_steps must be a comma-separated list of "
                "positive integers, for example: 5000,7000"
            ) from exc

        if not explicit_checkpoint_steps:
            raise ValueError(
                "checkpoint_steps was provided but no checkpoint "
                "steps were parsed."
            )

        if any(step <= 0 for step in explicit_checkpoint_steps):
            raise ValueError(
                "All checkpoint_steps values must be greater than zero."
            )

        if any(
            step > cfg.max_steps
            for step in explicit_checkpoint_steps
        ):
            raise ValueError(
                "All checkpoint_steps values must be less than or "
                "equal to max_steps."
            )

        print(
            "Saving checkpoints at explicit optimizer steps: "
            f"{sorted(explicit_checkpoint_steps)}"
        )
    else:
        if cfg.save_steps <= 0:
            raise ValueError(
                "save_steps must be greater than zero."
            )

        print(
            f"Saving checkpoints every {cfg.save_steps} "
            "optimizer steps."
        )

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto" if cfg.use_quantization else None,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
        # QLoRA memory optimization. 4-bit bitsandbytes models must not be moved with `.to()`.
        vla.config.use_cache = False
        if hasattr(vla, "gradient_checkpointing_enable"):
            # Non-reentrant checkpointing avoids DDP "marked ready twice" errors
            # with LoRA adapters on recent PyTorch versions.
            try:
                vla.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                vla.gradient_checkpointing_enable()
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Workaround for DDP + LoRA + checkpointing on a static graph. This avoids
    # "parameter has been marked as ready twice" failures during backward.
    if hasattr(vla, "_set_static_graph"):
        vla._set_static_graph()

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize W&B only if explicitly enabled in a future environment.
    if distributed_state.is_main_process and wandb is not None:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=f"ft+{exp_id}",
            mode=os.environ.get("WANDB_MODE", "online"),
        )

    # Deques for smoothed metrics across gradient-accumulation batches.
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    optimizer_step = 0

    # Train. max_steps counts optimizer updates, not micro-batches.
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"]
                    .to(torch.bfloat16)
                    .to(device_id),
                    labels=batch["labels"].to(device_id),
                )
                loss = output.loss

            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            action_logits = output.logits[
                :,
                vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1,
            ]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = (
                correct_preds.sum().float() / mask.sum().float()
            )

            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(
                    action_preds[mask].detach().cpu().numpy()
                )
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(
                    action_gt[mask].detach().cpu().numpy()
                )
            )
            action_l1_loss = torch.nn.functional.l1_loss(
                continuous_actions_pred,
                continuous_actions_gt,
            )

            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            should_step = (
                (batch_idx + 1) % cfg.grad_accumulation_steps == 0
            )

            if not should_step:
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            progress.update(1)

            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = (
                sum(recent_action_accuracies)
                / len(recent_action_accuracies)
            )
            smoothened_l1_loss = (
                sum(recent_l1_losses) / len(recent_l1_losses)
            )

            progress.set_postfix(
                loss=f"{smoothened_loss:.4f}",
                accuracy=f"{smoothened_action_accuracy:.4f}",
                l1=f"{smoothened_l1_loss:.4f}",
            )

            if (
                distributed_state.is_main_process
                and wandb is not None
                and optimizer_step % 10 == 0
            ):
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                    },
                    step=optimizer_step,
                )

            should_save = (
                optimizer_step in explicit_checkpoint_steps
                if explicit_checkpoint_steps is not None
                else (
                    optimizer_step > 0
                    and optimizer_step % cfg.save_steps == 0
                )
            )

            if should_save:
                if distributed_state.is_main_process:
                    print(
                        f"Saving Model Checkpoint for Step "
                        f"{optimizer_step}"
                    )

                    save_dir = (
                        adapter_dir if cfg.use_lora else run_dir
                    )

                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)

                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

                if cfg.use_lora:
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                    )

                    merged_vla = PeftModel.from_pretrained(
                        base_vla,
                        adapter_dir,
                    )
                    merged_vla = merged_vla.merge_and_unload()

                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:
                            merged_vla.save_pretrained(run_dir)
                            print(
                                "Saved Model Checkpoint for Step "
                                f"{optimizer_step} at: {run_dir}"
                            )
                        else:
                            checkpoint_dir = Path(
                                str(run_dir)
                                + f"--{optimizer_step}_chkpt"
                            )
                            checkpoint_dir.mkdir(
                                parents=True,
                                exist_ok=True,
                            )

                            save_dataset_statistics(
                                vla_dataset.dataset_statistics,
                                checkpoint_dir,
                            )
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)

                            print(
                                "Saved Model Checkpoint for Step "
                                f"{optimizer_step} at: "
                                f"{checkpoint_dir}"
                            )

                    del base_vla
                    del merged_vla
                    torch.cuda.empty_cache()

                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

            if optimizer_step >= cfg.max_steps:
                print(
                    f"Max step {cfg.max_steps} reached! "
                    "Stopping training..."
                )
                break



if __name__ == "__main__":
    finetune()
