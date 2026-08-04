# OpenVLA Fine-Tuning

This directory contains the project-specific `finetune.py` script used to fine-tune OpenVLA on the custom `ur5e_openvla` RLDS dataset.

The script is based on OpenVLA’s LoRA fine-tuning workflow and was adapted for the project’s RTX 4000 Ada workstation, QLoRA configuration, gradient accumulation, and checkpoint comparison process.

## Prerequisites

Before running fine-tuning:

1. Install OpenVLA by following the official [OpenVLA README](https://github.com/openvla/openvla).
2. Process the UR5e demonstrations.
3. Convert the processed episodes into the `ur5e_openvla` RLDS dataset.
4. Register the dataset with OpenVLA using:
   [`openvla_dataset_registration/`](../openvla_dataset_registration/)
5. Confirm that the RLDS dataset can be loaded and visualized successfully.

The dataset name must remain consistent across the RLDS builder, OpenVLA registration files, dataset directory, and training command:

```text
ur5e_openvla
```

## `finetune.py`

The script supports:

- LoRA and QLoRA fine-tuning;
- optional 4-bit NF4 quantization;
- gradient accumulation;
- `torchrun` and DistributedDataParallel;
- image augmentation;
- RLDS dataset loading;
- dataset-statistics saving;
- merged LoRA checkpoint saving;
- periodic or exact checkpoint-step selection;
- operation with Weights & Biases disabled.

## Dataset Registration

The RLDS dataset must be registered in OpenVLA’s:

```text
prismatic/vla/datasets/rlds/oxe/configs.py
prismatic/vla/datasets/rlds/oxe/transforms.py
```

The custom registration is documented in:

```text
openvla_dataset_registration/
```

The transform converts the raw RLDS gripper action from:

```text
0 = open
1 = closed
```

to OpenVLA’s expected action convention:

```text
1 = open
0 = closed
```

Only the gripper action target is inverted. The six relative translation and rotation dimensions remain unchanged.

## Dataset Path

`--data_root_dir` must point to the parent TensorFlow Datasets directory.

For example, if the dataset is stored at:

```text
/home/user/tensorflow_datasets/ur5e_openvla/2.1.0/
```

use:

```bash
--data_root_dir /home/user/tensorflow_datasets
--dataset_name ur5e_openvla
```

## Example Command

```bash
cd /path/to/openvla
conda activate openvla_finetune

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node 1 \
  /path/to/ur5e_hande_openVLA_integration/training/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/tensorflow_datasets \
  --dataset_name ur5e_openvla \
  --run_root_dir /path/to/training_runs \
  --adapter_tmp_dir /path/to/adapter_tmp \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 0.0001 \
  --max_steps 7000 \
  --checkpoint_steps 5000,7000 \
  --save_latest_checkpoint_only False \
  --use_lora True \
  --lora_rank 32 \
  --lora_dropout 0.0 \
  --use_quantization True \
  --image_aug True \
  --shuffle_buffer_size 4750 \
  --run_id_note red-block-only-7k-checkpoints
```

Replace each `/path/to/...` value with the correct local path.

## Important Arguments

| Argument | Purpose |
|---|---|
| `--data_root_dir` | Parent TensorFlow Datasets directory |
| `--dataset_name` | Registered RLDS dataset name |
| `--max_steps` | Number of optimizer updates |
| `--batch_size` | Micro-batch size |
| `--grad_accumulation_steps` | Micro-batches per optimizer update |
| `--checkpoint_steps` | Exact optimizer steps to save, such as `5000,7000` |
| `--save_steps` | Periodic save interval when exact steps are not provided |
| `--save_latest_checkpoint_only` | Overwrite one checkpoint or preserve numbered checkpoints |
| `--use_quantization` | Enable 4-bit QLoRA |
| `--image_aug` | Enable image augmentation |

The effective batch size is:

```text
batch_size × grad_accumulation_steps
```

For example:

```text
1 × 8 = effective batch size 8
```

## Checkpoint Saving

To save exact checkpoints:

```bash
--max_steps 7000 \
--checkpoint_steps 5000,7000 \
--save_latest_checkpoint_only False
```

This creates separately numbered directories for steps 5000 and 7000.

When `--checkpoint_steps` is provided, it overrides periodic saving through `--save_steps`.

Each saved checkpoint includes:

- the merged model;
- the processor;
- dataset statistics required for action de-normalization.

Fine-tuned checkpoints are large, so check available storage before training:

```bash
df -h
```

## Validation and Monitoring

Check the script before running:

```bash
python -m py_compile training/finetune.py
python training/finetune.py --help
```

Monitor GPU use in another terminal:

```bash
watch -n 2 nvidia-smi
```

## Current Use

Phase 4 uses a single-task dataset containing demonstrations for:

```text
Place the red block on the yellow platform.
```

Models are saved at selected checkpoints and compared using physical robot tests. After checkpoint evaluation, additional demonstrations will be added and new models will be trained on the expanded dataset.

## Related Directories

- [`data_processing/`](../data_processing/) cleans episodes and converts gripper deltas to absolute states.
- [`rlds_dataset_builder/`](../rlds_dataset_builder/) provides the custom RLDS builder.
- [`openvla_dataset_registration/`](../openvla_dataset_registration/) registers the dataset with OpenVLA.
- [`inference/`](../inference/) evaluates fine-tuned models on the UR5e.

## Notes

- Training is controlled by optimizer steps rather than conventional epochs.
- A merged checkpoint does not include optimizer state for exact training resumption.
- Checkpoint quality should be determined through physical evaluation, not training loss alone.
