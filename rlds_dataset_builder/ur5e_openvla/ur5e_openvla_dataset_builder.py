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

    VERSION = tfds.core.Version("2.0.0")
    RELEASE_NOTES = {
        "2.0.0": (
            "Changed the learned action to seven dimensions and replaced "
            "gripper delta with an absolute gripper target state."
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

    def _generate_examples(self, clean_root: Path) -> Iterator[Tuple[str, Any]]:
        """Yield one TFDS example per cleaned episode directory."""
        clean_root = Path(clean_root).expanduser()
        if not clean_root.exists():
            raise FileNotFoundError(
                f"Cleaned dataset root does not exist: {clean_root}\n"
                "Set UR5E_OPENVLA_CLEAN_ROOT=/path/to/ur5e_clean_absolute "
                "or update DEFAULT_CLEAN_DATA_ROOT."
            )

        episode_dirs = sorted(p for p in clean_root.glob("episode_*") if p.is_dir())
        if not episode_dirs:
            raise FileNotFoundError(f"No episode_* directories found under: {clean_root}")

        for episode_dir in episode_dirs:
            parsed = self._parse_episode(episode_dir)
            if parsed is None:
                continue
            yield parsed

    def _parse_episode(self, episode_dir: Path) -> Tuple[str, Any] | None:
        """Convert one cleaned episode folder into one RLDS-style TFDS sample."""
        complete_path = episode_dir / "COMPLETE.json"
        metadata_path = episode_dir / "episode_metadata.json"
        steps_path = episode_dir / "steps.jsonl"

        if not complete_path.exists() or not metadata_path.exists() or not steps_path.exists():
            return None

        complete = _read_json(complete_path)
        if not complete.get("cleaning_complete", False):
            # Only consume episodes generated by clean_raw_episodes.py.
            return None

        metadata = _read_json(metadata_path)
        episode_id = str(metadata.get("episode_id", episode_dir.name))
        task_id = str(metadata.get("task_id", ""))
        instruction = str(metadata.get("language_instruction", ""))
        source_episode_status = str(complete.get("source_episode_status", ""))
        language_embedding = self._embed([instruction])[0].numpy().astype(np.float32)

        raw_steps = list(_read_jsonl(steps_path))
        episode_steps = []

        for i, step in enumerate(raw_steps):
            converted = self._convert_step(
                episode_dir=episode_dir,
                step=step,
                instruction=instruction,
                language_embedding=language_embedding,
                step_index=i,
                num_steps=len(raw_steps),
            )
            if converted is None:
                # If cleaning worked correctly this should not happen, but skip
                # malformed steps rather than crashing a long conversion.
                continue
            episode_steps.append(converted)

        if not episode_steps:
            return None

        # Ensure flags are consistent after any malformed-step skipping.
        for i, step in enumerate(episode_steps):
            step["is_first"] = i == 0
            step["is_last"] = i == (len(episode_steps) - 1)
            step["is_terminal"] = i == (len(episode_steps) - 1)
            step["reward"] = float(i == (len(episode_steps) - 1))

        sample = {
            "steps": episode_steps,
            "episode_metadata": {
                "episode_id": episode_id,
                "task_id": task_id,
                "file_path": str(episode_dir),
                "source_episode_status": source_episode_status,
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
    ) -> dict[str, Any] | None:
        """Convert one cleaned JSONL step into the TFDS step schema."""
        image_rel_path = step.get("image", {}).get("path")
        if image_rel_path is None:
            return None
        image_path = episode_dir / image_rel_path
        if not image_path.exists():
            return None

        tcp_pose = step.get("follower", {}).get("tcp_pose_actual")
        gripper_closed = step.get("gripper", {}).get("closed")
        source_action = step.get("action_to_next_raw", {}).get("action_8d_raw")

        if tcp_pose is None or gripper_closed is None or source_action is None:
            return None
        if len(tcp_pose) != 6 or len(source_action) != 8:
            return None

        state = np.asarray(list(tcp_pose) + [gripper_closed], dtype=np.float32)

        # The converted cleaned dataset retains an eighth termination value for
        # traceability. Termination remains represented separately through
        # is_last, is_terminal, and reward, so the learned action is 7D.
        action = np.asarray(source_action[:7], dtype=np.float32)

        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
            return None

        return {
            "observation": {
                # TFDS Image features accept an image file path and encode it.
                "image": str(image_path),
                "state": state,
            },
            "action": action,
            "discount": np.float32(1.0),
            "reward": np.float32(step_index == (num_steps - 1)),
            "is_first": bool(step.get("is_first", step_index == 0)),
            "is_last": bool(step.get("is_last", step_index == (num_steps - 1))),
            "is_terminal": bool(step.get("is_terminal", step_index == (num_steps - 1))),
            "language_instruction": instruction,
            "language_embedding": language_embedding,
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
