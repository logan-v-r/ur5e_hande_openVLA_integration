"""
Load the fine-tuned UR5e OpenVLA model and produce processed actions.

The deployed fine-tuned `ur5e_openvla` model produces seven-dimensional actions:

    [dx, dy, dz, drx, dry, drz, gripper_delta]

Action conventions
------------------

Translation:
    [dx, dy, dz] is a relative Cartesian displacement expressed in the
    UR5e base frame, in metres.

Rotation:
    [drx, dry, drz] is a relative base-frame rotation vector. The vector
    direction defines the rotation axis, and its magnitude defines the
    rotation angle in radians.

    These values are already a rotation-vector representation. They must
    not be interpreted as roll, pitch, and yaw or converted from Euler
    angles.

Gripper:
    The seventh value is the change in binary gripper closed state:

        positive -> move toward closed
        negative -> move toward open
        near zero -> no state change

This module handles model loading, image preparation, action
de-normalization, and action formatting. It does not calculate a UR5e
target pose or command robot hardware.

UR5e target calculation is handled by `ur5_action_adapter.py`.
Physical execution is handled by `openvla_move_with_liveview.py`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


ACTION_DIMENSION = 7

ACTION_DIM_LABELS = [
    "dx",
    "dy",
    "dz",
    "drx",
    "dry",
    "drz",
    "gripper_delta",
]


class OpenVLAInference:
    """
    Run single-action inference with the fine-tuned UR5e OpenVLA model.

    One RGB observation and one language instruction produce one
    seven-dimensional action. The physical result of that action should
    be observed in a new image before the next call to `step()`.
    """

    def __init__(
        self,
        saved_model_path: str,
        unnorm_key: Optional[str] = None,
        action_scale: float = 1.0,
    ) -> None:
        """
        Load the fine-tuned model and its dataset normalization statistics.

        Args:
            saved_model_path:
                Local path to the merged fine-tuned OpenVLA model
                directory.

            unnorm_key:
                Dataset-statistics key used by `predict_action()` to
                de-normalize the model output. When omitted, the class
                selects `ur5e_openvla` when available or automatically
                selects the only available statistics key.

            action_scale:
                Optional multiplier applied to translation and rotation
                deltas after de-normalization. The gripper value is not scaled.

        Raises:
            ValueError:
                If arguments are invalid.

            RuntimeError:
                If CUDA or model normalization statistics are unavailable.

            FileNotFoundError:
                If a local fine-tuned model directory does not contain
                `dataset_statistics.json`.
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if not saved_model_path.strip():
            raise ValueError("saved_model_path cannot be empty.")

        if not np.isfinite(action_scale) or action_scale <= 0.0:
            raise ValueError(
                "action_scale must be a finite value greater than zero."
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "OpenVLA inference requires a CUDA-enabled PyTorch "
                "installation."
            )

        self.saved_model_path = saved_model_path
        self.action_scale = float(action_scale)
        self.device = torch.device("cuda:0")

        self.task_description: Optional[str] = None
        self.num_image_history = 0

        print(f"Loading processor from: {saved_model_path}")

        self.processor = AutoProcessor.from_pretrained(
            saved_model_path,
            trust_remote_code=True,
        )

        print(f"Loading model from: {saved_model_path}")

        self.vla = AutoModelForVision2Seq.from_pretrained(
            saved_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        self.vla.eval()

        self._load_local_dataset_statistics()
        self.unnorm_key = self._resolve_unnorm_key(unnorm_key)

        print("OpenVLA model loaded.")
        print(f"Normalization key: {self.unnorm_key}")
        print(f"Expected action dimensions: {ACTION_DIMENSION}")
        print(f"Translation/rotation action scale: {self.action_scale}")

    def _load_local_dataset_statistics(self) -> None:
        """
        Load dataset statistics saved by the fine-tuning script.

        OpenVLA requires the training dataset statistics to de-normalize
        predicted actions. For a local merged fine-tuned model, these
        statistics are expected in:

            <saved_model_path>/dataset_statistics.json

        If `saved_model_path` is not a local directory, this method leaves
        any normalization statistics already attached to the model
        unchanged.
        """
        model_path = Path(self.saved_model_path).expanduser()

        if not model_path.is_dir():
            return

        statistics_path = model_path / "dataset_statistics.json"

        if not statistics_path.exists():
            raise FileNotFoundError(
                "The local fine-tuned model directory does not contain "
                f"dataset_statistics.json: {statistics_path}"
            )

        with statistics_path.open("r", encoding="utf-8") as file:
            statistics = json.load(file)

        if not isinstance(statistics, dict) or not statistics:
            raise ValueError(
                f"Invalid or empty dataset statistics: {statistics_path}"
            )

        self.vla.norm_stats = statistics

    def _resolve_unnorm_key(
        self,
        requested_key: Optional[str],
    ) -> str:
        """
        Validate or infer the dataset normalization key.
        """
        norm_stats = getattr(self.vla, "norm_stats", None)

        if not isinstance(norm_stats, dict) or not norm_stats:
            raise RuntimeError(
                "The loaded model does not contain dataset normalization "
                "statistics. Ensure dataset_statistics.json from the "
                "fine-tuning run is available in the model directory."
            )

        available_keys = list(norm_stats.keys())

        if requested_key is not None:
            if requested_key not in norm_stats:
                raise KeyError(
                    f"Normalization key {requested_key!r} was not found. "
                    f"Available keys: {available_keys}"
                )

            return requested_key

        if "ur5e_openvla" in norm_stats:
            return "ur5e_openvla"

        if len(available_keys) == 1:
            return available_keys[0]

        raise KeyError(
            "Multiple dataset normalization keys are available and none "
            "was specified. Pass `unnorm_key` explicitly. Available keys: "
            f"{available_keys}"
        )

    def reset(self, task_description: str) -> None:
        """
        Reset episode-level state for a new language instruction.
        """
        instruction = task_description.strip()

        if not instruction:
            raise ValueError("task_description cannot be empty.")

        self.task_description = instruction
        self.num_image_history = 0

    def step(
        self,
        image: np.ndarray,
        task_description: Optional[str] = None,
        *args,
        **kwargs,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Predict and format one action from the fine-tuned UR5e model.

        Args:
            image:
                RGB uint8 image with shape [height, width, 3].

            task_description:
                Natural-language instruction. Passing a different
                instruction resets the episode-level state.

        Returns:
            raw_action:
                De-normalized model output divided into:

                    world_vector
                    rotation_delta_base_frame
                    gripper_delta

            action:
                Processed action consumed by `ur5_action_adapter.py`:

                    world_vector
                    rot_axangle
                    gripper

        Notes:
            The rotation output is already a base-frame rotation vector.
            No Euler-angle conversion is performed.
        """
        del args, kwargs

        if (
            task_description is not None
            and task_description != self.task_description
        ):
            self.reset(task_description)

        instruction = task_description or self.task_description

        if instruction is None or not instruction.strip():
            raise ValueError(
                "No task description is available. Call reset(...) first "
                "or pass task_description to step(...)."
            )

        self._validate_image(image)

        prompt = (
            "In: What action should the robot take to "
            f"{instruction.strip().lower()}?\n"
            "Out:"
        )

        pil_image = Image.fromarray(image).convert("RGB")

        inputs = self.processor(
            prompt,
            pil_image,
        ).to(
            self.device,
            dtype=torch.bfloat16,
        )

        with torch.inference_mode():
            predicted_action = self.vla.predict_action(
                **inputs,
                unnorm_key=self.unnorm_key,
                do_sample=False,
            )

        predicted_action = np.asarray(
            predicted_action,
            dtype=np.float64,
        ).reshape(-1)

        self._validate_predicted_action(predicted_action)

        raw_action = {
            "world_vector": predicted_action[0:3].copy(),
            "rotation_delta_base_frame": predicted_action[3:6].copy(),
            "gripper_delta": predicted_action[6:7].copy(),
        }

        action = {
            "world_vector": (
                raw_action["world_vector"] * self.action_scale
            ),
            "rot_axangle": (
                raw_action["rotation_delta_base_frame"]
                * self.action_scale
            ),
            "gripper": raw_action["gripper_delta"].copy(),
        }

        self.num_image_history += 1

        return raw_action, action

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """
        Validate one RGB camera observation.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"image must be a NumPy array, got {type(image).__name__}."
            )

        if image.dtype != np.uint8:
            raise ValueError(
                f"image must use dtype uint8, got {image.dtype}."
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "image must have shape [height, width, 3], "
                f"got {image.shape}."
            )

        if image.size == 0:
            raise ValueError("image cannot be empty.")

    @staticmethod
    def _validate_predicted_action(
        predicted_action: np.ndarray,
    ) -> None:
        """
        Validate the de-normalized action returned by OpenVLA.
        """
        if predicted_action.size != ACTION_DIMENSION:
            raise ValueError(
                "The fine-tuned model returned an unexpected action "
                f"size. Expected {ACTION_DIMENSION}, received "
                f"{predicted_action.size}: {predicted_action}"
            )

        if not np.all(np.isfinite(predicted_action)):
            raise ValueError(
                "The fine-tuned model returned non-finite action values: "
                f"{predicted_action}"
            )

    def visualize_epoch(
        self,
        predicted_raw_actions: Sequence[dict[str, np.ndarray]],
        images: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        """
        Save a visualization of images and predicted seven-dimensional
        actions from one evaluation episode.
        """
        if not predicted_raw_actions:
            raise ValueError("predicted_raw_actions cannot be empty.")

        if not images:
            raise ValueError("images cannot be empty.")

        display_images = [
            np.asarray(
                Image.fromarray(image).resize((224, 224))
            )
            for image in images
        ]

        sampled_images = display_images[::3]

        if not sampled_images:
            sampled_images = [display_images[0]]

        image_strip = np.concatenate(sampled_images, axis=1)

        predicted_actions = np.asarray(
            [
                np.concatenate(
                    [
                        action["world_vector"],
                        action["rotation_delta_base_frame"],
                        action["gripper_delta"],
                    ]
                )
                for action in predicted_raw_actions
            ],
            dtype=np.float64,
        )

        if predicted_actions.ndim != 2:
            raise ValueError(
                "Predicted actions could not be converted to a "
                "two-dimensional array."
            )

        if predicted_actions.shape[1] != ACTION_DIMENSION:
            raise ValueError(
                "Expected visualized actions to contain "
                f"{ACTION_DIMENSION} values, got "
                f"{predicted_actions.shape[1]}."
            )

        figure_layout = [
            ["image"] * len(ACTION_DIM_LABELS),
            ACTION_DIM_LABELS,
        ]

        plt.rcParams.update({"font.size": 12})

        figure, axes = plt.subplot_mosaic(figure_layout)
        figure.set_size_inches(45, 10)

        axes["image"].imshow(image_strip)
        axes["image"].set_xlabel(
            "Time in one episode (subsampled)"
        )
        axes["image"].set_xticks([])
        axes["image"].set_yticks([])

        for action_index, action_label in enumerate(
            ACTION_DIM_LABELS
        ):
            axes[action_label].plot(
                predicted_actions[:, action_index],
                label="Predicted action",
            )
            axes[action_label].set_title(action_label)
            axes[action_label].set_xlabel("Inference step")
            axes[action_label].legend()

        figure.tight_layout()
        figure.savefig(save_path)
        plt.close(figure)
