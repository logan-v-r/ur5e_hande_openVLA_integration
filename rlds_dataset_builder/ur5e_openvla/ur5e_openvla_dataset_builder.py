"""TFDS/RLDS builder for the cleaned UR5e OpenVLA dataset.

This builder converts the cleaned episode folders produced by
`clean_raw_episodes.py` into an RLDS-style TFDS dataset.

Expected cleaned episode layout:

    ~/workspaces/openvla/datasets/ur5e_clean_absolute/
      episode_.../
        episode_metadata.json
        steps.jsonl
        COMPLETE.json
        images/000000.jpg
        images/000001.jpg
        ...

Run from this directory with:

    tfds build

Optional: override the cleaned-data folder without editing this file:

    UR5E_OPENVLA_CLEAN_ROOT=/path/to/ur5e_clean_absolute tfds build
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


# Default cleaned-data location. You can override this with the environment
# variable UR5E_OPENVLA_CLEAN_ROOT when running `tfds build`.
DEFAULT_CLEAN_DATA_ROOT = (
    Path.home() / "workspaces" / "openvla" / "datasets" / "ur5e_clean_absolute"
)


class Ur5eOpenvla(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cleaned UR5e + RealSense OpenVLA demonstrations.

    Each example is one episode. Each episode contains a sequence of steps with:
      - observation.image: native RGB camera frame from RealSense
      - observation.state: [x, y, z, rx, ry, rz, gripper_closed]
      - action: [dx, dy, dz, drx, dry, drz, gripper_closed]
        where gripper_closed is an absolute target: 0.0=open, 1.0=closed
      - language_instruction: task instruction repeated per step
    """

    VERSION = tfds.core.Version("2.1.0")
    RELEASE_NOTES = {
        "2.0.0": (
            "Changed the learned action to seven dimensions and replaced "
            "gripper delta with an absolute gripper target state."
        ),
        "2.1.0": (
            "Standardized episode language instructions and added strict "
            "validation of required files, images, state, and absolute-gripper "
            "actions."
        ),
    }

    def __init__(self, *args, **kwargs):
        """Initialize the builder and the language embedding model."""
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        self._clean_root = Path(os.environ.get("UR5E_OPENVLA_CLEAN_ROOT", DEFAULT_CLEAN_DATA_ROOT)).expanduser()

    def _info(self) -> tfds.core.DatasetInfo:
        """Define the RLDS/TFDS feature schema produced by this builder."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(None, None, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="Main RealSense RGB observation. Stored at native cleaned-image resolution.",
                        ),
                        "state": tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc=(
                                "UR5e state [x, y, z, rx, ry, rz, gripper_closed]. "
                                "TCP pose is in the UR5e base frame; gripper_closed is 0=open, 1=closed."
                            ),
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc=(
                            "Action [dx, dy, dz, drx, dry, drz, gripper_closed]. "
                            "TCP delta is in the UR5e base frame. "
                            "gripper_closed is an absolute target state: "
                            "0.0=open and 1.0=closed."
                        ),
                    ),
                    "discount": tfds.features.Scalar(
                        dtype=np.float32,
                        doc="Discount for demos; set to 1.0 for all steps.",
                    ),
                    "reward": tfds.features.Scalar(
                        dtype=np.float32,
                        doc="Sparse demo reward; 1.0 on the final step, otherwise 0.0.",
                    ),
                    "is_first": tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="True on the first step of the episode.",
                    ),
                    "is_last": tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="True on the last step of the episode.",
                    ),
                    "is_terminal": tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="True on the terminal final step.",
                    ),
                    "language_instruction": tfds.features.Text(
                        doc="Language instruction for the episode, repeated at each step.",
                    ),
                    "language_embedding": tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc="Universal Sentence Encoder embedding of the language instruction.",
                    ),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "episode_id": tfds.features.Text(doc="Cleaned episode ID."),
                    "task_id": tfds.features.Text(doc="Task identifier from episode_metadata.json."),
                    "file_path": tfds.features.Text(doc="Path to the cleaned episode directory."),
                    "source_episode_status": tfds.features.Text(doc="Raw episode status copied through cleaning."),
                }),
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define dataset splits.

        The cleaned-data folder is treated as the train split by default. For a
        first dataset, keep held-out/OOD tests outside this folder or maintain a
        separate cleaned folder for evaluation.
        """
        return {
            "train": self._generate_examples(self._clean_root),
        }

    def _generate_examples(
        self,
        clean_root: Path,
    ) -> Iterator[Tuple[str, Any]]:
        """Yield one validated TFDS example per cleaned episode directory."""
        clean_root = Path(clean_root).expanduser()

        if not clean_root.exists():
            raise FileNotFoundError(
                f"Cleaned dataset root does not exist: {clean_root}\n"
                "Set UR5E_OPENVLA_CLEAN_ROOT=/path/to/ur5e_clean_absolute "
                "or update DEFAULT_CLEAN_DATA_ROOT."
            )

        episode_dirs = sorted(
            path
            for path in clean_root.glob("episode_*")
            if path.is_dir()
        )

        if not episode_dirs:
            raise FileNotFoundError(
                f"No episode_* directories found under: {clean_root}"
            )

        for episode_dir in episode_dirs:
            yield self._parse_episode(episode_dir)

    def _parse_episode(
        self,
        episode_dir: Path,
    ) -> Tuple[str, Any]:
        """Convert one validated cleaned episode into one RLDS sample."""
        complete_path = episode_dir / "COMPLETE.json"
        metadata_path = episode_dir / "episode_metadata.json"
        steps_path = episode_dir / "steps.jsonl"

        for required_path in (
            complete_path,
            metadata_path,
            steps_path,
        ):
            if not required_path.exists():
                raise FileNotFoundError(
                    f"{episode_dir.name}: required file is missing: "
                    f"{required_path}"
                )

        complete = _read_json(complete_path)

        if not complete.get("cleaning_complete", False):
            raise ValueError(
                f"{episode_dir.name}: COMPLETE.json does not report "
                "cleaning_complete=true"
            )

        metadata = _read_json(metadata_path)

        episode_id = str(
            metadata.get(
                "episode_id",
                episode_dir.name,
            )
        )

        task_id = str(
            metadata.get(
                "task_id",
                "",
            )
        )

        # The authoritative instruction is stored once per cleaned episode in
        # episode_metadata.json. It is repeated into every RLDS step below.
        instruction_value = metadata.get(
            "language_instruction"
        )

        if not isinstance(instruction_value, str):
            raise ValueError(
                f"{episode_dir.name}: episode_metadata.json does not contain "
                "a string language_instruction"
            )

        instruction = instruction_value.strip()

        if not instruction:
            raise ValueError(
                f"{episode_dir.name}: language_instruction is empty"
            )

        source_episode_status = str(
            complete.get(
                "source_episode_status",
                "",
            )
        )

        language_embedding = (
            self._embed([instruction])[0]
            .numpy()
            .astype(np.float32)
        )

        raw_steps = list(
            _read_jsonl(steps_path)
        )

        if not raw_steps:
            raise ValueError(
                f"{episode_dir.name}: steps.jsonl contains no steps"
            )

        episode_steps = [
            self._convert_step(
                episode_dir=episode_dir,
                step=step,
                instruction=instruction,
                language_embedding=language_embedding,
                step_index=step_index,
                num_steps=len(raw_steps),
            )
            for step_index, step in enumerate(raw_steps)
        ]

        sample = {
            "steps": episode_steps,
            "episode_metadata": {
                "episode_id": episode_id,
                "task_id": task_id,
                "file_path": str(episode_dir),
                "source_episode_status":
                    source_episode_status,
            },
        }

        return episode_dir.name, sample

    def _convert_step(
        self,
        episode_dir: Path,
        step: dict[str, Any],
        instruction: str,
        language_embedding: np.ndarray,
        step_index: int,
        num_steps: int,
    ) -> dict[str, Any]:
        """Convert one validated cleaned JSONL step into the TFDS schema."""
        image_data = step.get("image")

        if not isinstance(image_data, dict):
            raise ValueError(
                f"{episode_dir.name} step {step_index}: "
                "missing image object"
            )

        image_rel_path = image_data.get("path")

        if (
            not isinstance(image_rel_path, str)
            or not image_rel_path.strip()
        ):
            raise ValueError(
                f"{episode_dir.name} step {step_index}: "
                "missing image.path"
            )

        image_path = episode_dir / image_rel_path

        if not image_path.exists():
            raise FileNotFoundError(
                f"{episode_dir.name} step {step_index}: "
                f"image does not exist: {image_path}"
            )

        follower = step.get("follower")

        if not isinstance(follower, dict):
            raise ValueError(
                f"{episode_dir.name} step {step_index}: "
                "missing follower object"
            )

        tcp_pose = follower.get(
            "tcp_pose_actual"
        )

        if (
            not isinstance(tcp_pose, (list, tuple))
            or len(tcp_pose) != 6
        ):
            raise ValueError(
                f"{episode_dir.name} step {step_index}: "
                "expected follower.tcp_pose_actual to contain 6 values"
            )

        gripper_data = step.get("gripper")

        if not isinstance(gripper_data, dict):
            raise ValueError(
                f"{episode_dir.name} step {step_index}: "
                "missing gripper object"
            )

        gripper_closed = gripper_data.get(
            "closed"
        )

        action_data = step.get(
            "action_to_next_raw"
        )

        if not isinstance(action_data, dict):
            raise ValueError(
                f"{episode_dir.name} step {step_index}: "
                "missing action_to_next_raw object"
            )

        tcp_delta = action_data.get(
            "tcp_delta_base_frame"
        )

        gripper_target = action_data.get(
            "gripper_absolute_binary"
        )

        representation = action_data.get(
            "gripper_action_representation"
        )

        if representation != "absolute_target_state":
            raise ValueError(
                f"{episode_dir.name} step {step_index}: expected "
                "gripper_action_representation='absolute_target_state', "
                f"got {representation!r}"
            )

        if (
            not isinstance(tcp_delta, (list, tuple))
            or len(tcp_delta) != 6
        ):
            raise ValueError(
                f"{episode_dir.name} step {step_index}: "
                "expected tcp_delta_base_frame to contain 6 values"
            )

        if gripper_closed not in (0, 0.0, 1, 1.0):
            raise ValueError(
                f"{episode_dir.name} step {step_index}: "
                f"invalid gripper.closed value: {gripper_closed!r}"
            )

        if gripper_target not in (0, 0.0, 1, 1.0):
            raise ValueError(
                f"{episode_dir.name} step {step_index}: invalid "
                "gripper_absolute_binary value: "
                f"{gripper_target!r}"
            )

        state = np.asarray(
            list(tcp_pose) + [gripper_closed],
            dtype=np.float32,
        )

        # Construct the learned action from the explicit semantic fields.
        action = np.asarray(
            list(tcp_delta) + [gripper_target],
            dtype=np.float32,
        )

        if state.shape != (7,):
            raise ValueError(
                f"{episode_dir.name} step {step_index}: "
                f"state shape is {state.shape}, expected (7,)"
            )

        if action.shape != (7,):
            raise ValueError(
                f"{episode_dir.name} step {step_index}: "
                f"action shape is {action.shape}, expected (7,)"
            )

        if not np.all(np.isfinite(state)):
            raise ValueError(
                f"{episode_dir.name} step {step_index}: "
                "state contains non-finite values"
            )

        if not np.all(np.isfinite(action)):
            raise ValueError(
                f"{episode_dir.name} step {step_index}: "
                "action contains non-finite values"
            )

        # action_8d_raw is retained for traceability. Check it when present,
        # but do not require it for future episodes.
        trace_action = action_data.get(
            "action_8d_raw"
        )

        if trace_action is not None:
            if (
                not isinstance(trace_action, (list, tuple))
                or len(trace_action) != 8
            ):
                raise ValueError(
                    f"{episode_dir.name} step {step_index}: "
                    "action_8d_raw must contain 8 values when present"
                )

            trace_learned_action = np.asarray(
                trace_action[:7],
                dtype=np.float32,
            )

            if not np.allclose(
                action,
                trace_learned_action,
                atol=1e-7,
                rtol=0.0,
            ):
                raise ValueError(
                    f"{episode_dir.name} step {step_index}: "
                    "explicit 7D action does not match "
                    "action_8d_raw[:7]"
                )

            terminate = float(
                trace_action[7]
            )

            if terminate not in (0.0, 1.0):
                raise ValueError(
                    f"{episode_dir.name} step {step_index}: "
                    f"invalid termination trace value: {terminate}"
                )

        is_first = step_index == 0
        is_last = step_index == (
            num_steps - 1
        )

        return {
            "observation": {
                "image": str(image_path),
                "state": state,
            },
            "action": action,
            "discount": np.float32(1.0),
            "reward": np.float32(is_last),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_last,
            "language_instruction": instruction,
            "language_embedding":
                language_embedding,
        }


def _read_json(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one JSON object per non-empty line in a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
