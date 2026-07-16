#!/usr/bin/env python3
"""
clean_raw_episodes.py
=====================
Clean raw UR5e/OpenVLA demonstration episodes without modifying the raw data.

Input:
  raw_root/episode_*/episode_metadata.json
  raw_root/episode_*/steps.jsonl
  raw_root/episode_*/COMPLETE.json
  raw_root/episode_*/images/*.jpg

Output:
  cleaned_root/episode_*/episode_metadata.json
  cleaned_root/episode_*/steps.jsonl
  cleaned_root/episode_*/COMPLETE.json
  cleaned_root/episode_*/images/*.jpg
  cleaned_root/cleaning_manifest.jsonl

The script removes unusable steps, drops long near-zero-action runs, keeps a small
amount of idle context around meaningful actions, copies only retained images,
reindexes steps/images, and recomputes actions between retained steps.
Images are NOT resized here; keep native RGB files and let the RLDS/OpenVLA
preprocessing pipeline resize/crop/normalize them consistently.

Example usage
-------------

Run from the repository root:

python data_processing/clean_raw_episodes.py \
    --input-root /path/to/raw_episodes \
    --output-root /path/to/cleaned_episodes

For example:

python data_processing/clean_raw_episodes.py \
    --input-root ~/openvla/datasets/raw \
    --output-root ~/openvla/datasets/cleaned

By default, the script:
* requires each episode to contain a COMPLETE.json marker;
* rejects steps with invalid camera, TCP, safety, timing, or image data;
* treats translation of at least 0.002 m, rotation of at least 1 degree,
  or a gripper delta of at least 0.5 as meaningful action;
* keeps one idle context step before and after meaningful movement;
* rejects cleaned episodes with fewer than five retained steps;
* adds a terminal action to the final retained step; and
* does not overwrite an existing cleaned episode.

To replace existing cleaned output, add:

--overwrite

Example with selected custom thresholds:

python data_processing/clean_raw_episodes.py \
    --input-root ~/openvla/datasets/raw \
    --output-root ~/openvla/datasets/cleaned \
    --min-translation-m 0.002 \
    --min-rotation-deg 1.0 \
    --keep-idle-before-motion 1 \
    --keep-idle-after-motion 1 \
    --max-dt-s 0.6 \
    --min-steps 5 \
    --overwrite

Run the following command to view every available option:

python data_processing/clean_raw_episodes.py --help

"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R


def utc_now() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_complete_json(episode_dir: Path) -> dict[str, Any]:
    """Read an episode COMPLETE.json file when present, otherwise return {}."""
    complete_path = episode_dir / "COMPLETE.json"
    if not complete_path.exists():
        return {}
    return read_json(complete_path)


def write_json(path: Path, value: Any) -> None:
    """Write readable JSON."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON steps."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write newline-delimited JSON steps."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + "\n")


def nested_get(record: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    """Safely read nested dictionary values."""
    cur: Any = record
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def tcp_delta(pose_a: list[float], pose_b: list[float]) -> list[float]:
    """Return [dx, dy, dz, drx, dry, drz] between two TCP poses."""
    a = np.asarray(pose_a, dtype=np.float64)
    b = np.asarray(pose_b, dtype=np.float64)
    translation = b[:3] - a[:3]
    rotation = (R.from_rotvec(b[3:6]) * R.from_rotvec(a[3:6]).inv()).as_rotvec()
    return np.concatenate((translation, rotation)).tolist()


def action_magnitudes(delta: list[float]) -> tuple[float, float]:
    """Return translation norm in meters and rotation norm in radians."""
    return (
        float(np.linalg.norm(np.asarray(delta[:3], dtype=np.float64))),
        float(np.linalg.norm(np.asarray(delta[3:6], dtype=np.float64))),
    )


def gripper_delta(step_a: dict[str, Any], step_b: dict[str, Any]) -> Optional[float]:
    """Return binary gripper change, or None if either state is missing."""
    a = nested_get(step_a, ["gripper", "closed"])
    b = nested_get(step_b, ["gripper", "closed"])
    if a is None or b is None:
        return None
    return float(b) - float(a)


def has_meaningful_action(
    delta: list[float],
    grip_delta: Optional[float],
    min_translation_m: float,
    min_rotation_rad: float,
    min_gripper_delta: float,
) -> bool:
    """Return True when arm movement or gripper change is large enough to keep."""
    translation_m, rotation_rad = action_magnitudes(delta)
    arm_moved = translation_m >= min_translation_m or rotation_rad >= min_rotation_rad
    gripper_changed = grip_delta is not None and abs(grip_delta) >= min_gripper_delta
    return arm_moved or gripper_changed


@dataclass
class CleanConfig:
    """Cleaning configuration supplied by the command line."""

    input_root: Path
    output_root: Path
    overwrite: bool = False
    min_translation_m: float = 0.002
    min_rotation_deg: float = 1.0
    min_gripper_delta: float = 0.5
    keep_idle_before_motion: int = 1
    keep_idle_after_motion: int = 1
    max_dt_s: Optional[float] = 0.6
    min_steps: int = 5
    require_complete: bool = True
    require_camera_ok: bool = True
    require_tcp_available: bool = True
    require_safety_ok: bool = True
    require_image_file: bool = True
    terminal_action: bool = True


class EpisodeCleaner:
    """Clean raw episodes and write new cleaned episode folders."""

    def __init__(self, cfg: CleanConfig):
        self.cfg = cfg
        self.min_rotation_rad = np.deg2rad(cfg.min_rotation_deg)

    def discover_episodes(self) -> list[Path]:
        """Find episode directories directly under the input root."""
        root = self.cfg.input_root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Input root does not exist: {root}")
        return [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "steps.jsonl").exists()]

    def clean_all(self) -> None:
        """Clean every episode and write one manifest row per episode."""
        output_root = self.cfg.output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_rows: list[dict[str, Any]] = []

        for episode_dir in self.discover_episodes():
            try:
                summary = self.clean_episode(episode_dir, output_root)
            except Exception as exc:
                summary = {
                    "episode_id": episode_dir.name,
                    "input_dir": str(episode_dir.resolve()),
                    "output_dir": None,
                    "status": "error",
                    "error": str(exc),
                    "raw_steps": 0,
                    "kept_steps": 0,
                    "rejection_counts": {},
                }
            manifest_rows.append(summary)
            print(f"{summary['episode_id']}: {summary['status']} ({summary['kept_steps']}/{summary['raw_steps']} kept)")

        write_jsonl(output_root / "cleaning_manifest.jsonl", manifest_rows)
        print(f"\nWrote manifest: {output_root / 'cleaning_manifest.jsonl'}")

    def clean_episode(self, raw_dir: Path, output_root: Path) -> dict[str, Any]:
        """Clean one raw episode directory."""
        episode_id = raw_dir.name
        out_dir = output_root / episode_id
        out_images = out_dir / "images"
        raw_steps = load_jsonl(raw_dir / "steps.jsonl")
        source_complete = read_complete_json(raw_dir)

        if self.cfg.require_complete:
            if not source_complete:
                return self.summary(episode_id, raw_dir, "skipped_missing_complete_json", len(raw_steps), 0)
            if source_complete.get("recording_complete") is not True:
                return self.summary(episode_id, raw_dir, "skipped_recording_not_complete", len(raw_steps), 0)
        if out_dir.exists():
            if not self.cfg.overwrite:
                return self.summary(episode_id, raw_dir, "skipped_output_exists", len(raw_steps), 0, out_dir)
            shutil.rmtree(out_dir)

        candidates, rejection_counts = self.select_candidate_steps(raw_dir, raw_steps)
        cleaned = self.reindex_and_recompute(raw_dir, candidates)

        if len(cleaned) < self.cfg.min_steps:
            return self.summary(
                episode_id, raw_dir, "skipped_too_few_steps", len(raw_steps), len(cleaned), None, rejection_counts
            )

        out_images.mkdir(parents=True, exist_ok=True)
        for step in cleaned:
            src_image = Path(step.pop("_source_image_abs"))
            dst_image = out_dir / step["image"]["path"]
            shutil.copy2(src_image, dst_image)

        write_jsonl(out_dir / "steps.jsonl", cleaned)
        metadata = read_json(raw_dir / "episode_metadata.json") if (raw_dir / "episode_metadata.json").exists() else {}
        metadata["cleaning"] = {
            "cleaned_utc": utc_now(),
            "source_episode_dir": str(raw_dir.resolve()),
            "source_complete": source_complete,
            "source_episode_status": source_complete.get("episode_status"),
            "raw_steps": len(raw_steps),
            "raw_steps_written": source_complete.get("steps_written"),
            "raw_images_written": source_complete.get("images_written"),
            "kept_steps": len(cleaned),
            "rejection_counts": rejection_counts,
            "thresholds": {
                "min_translation_m": self.cfg.min_translation_m,
                "min_rotation_deg": self.cfg.min_rotation_deg,
                "min_gripper_delta": self.cfg.min_gripper_delta,
                "keep_idle_before_motion": self.cfg.keep_idle_before_motion,
                "keep_idle_after_motion": self.cfg.keep_idle_after_motion,
                "max_dt_s": self.cfg.max_dt_s,
            },
            "image_note": "Images remain at native collection resolution; resizing is deferred to RLDS/OpenVLA preprocessing.",
        }
        write_json(out_dir / "episode_metadata.json", metadata)
        write_json(
            out_dir / "COMPLETE.json",
            {
                "recording_complete": True,
                "cleaning_complete": True,
                "closed_utc": utc_now(),
                "source_episode_dir": str(raw_dir.resolve()),
                "source_episode_status": source_complete.get("episode_status"),
                "raw_steps_written": source_complete.get("steps_written", len(raw_steps)),
                "raw_images_written": source_complete.get("images_written"),
                "steps_written": len(cleaned),
                "images_written": len(cleaned),
                "episode_status": "cleaned_unreviewed",
            },
        )
        return self.summary(episode_id, raw_dir, "cleaned", len(raw_steps), len(cleaned), out_dir, rejection_counts)

    def select_candidate_steps(
        self, raw_dir: Path, raw_steps: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Apply quality gates and keep meaningful actions plus small idle context.

        A few still frames at the beginning/end of a demonstration are useful
        visual context. Long still stretches are harmful, so this keeps every
        meaningful-action step and a configurable number of adjacent idle steps.
        """
        rejection_counts: dict[str, int] = {}

        def reject(reason: str) -> None:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        valid_rows: list[tuple[int, dict[str, Any], bool]] = []

        for original_index, step in enumerate(raw_steps):
            reason = self.basic_rejection_reason(raw_dir, step)
            if reason is not None:
                reject(reason)
                continue

            action = step.get("action_to_next_raw")
            if not isinstance(action, dict):
                reject("missing_action")
                continue

            delta = action.get("tcp_delta_base_frame")
            if not isinstance(delta, list) or len(delta) != 6:
                reject("missing_tcp_delta")
                continue

            if self.cfg.max_dt_s is not None:
                dt_s = action.get("dt_s")
                if dt_s is None or float(dt_s) <= 0 or float(dt_s) > self.cfg.max_dt_s:
                    reject("bad_dt")
                    continue

            grip_delta = action.get("gripper_delta_binary")
            meaningful = has_meaningful_action(
                delta, grip_delta, self.cfg.min_translation_m, self.min_rotation_rad, self.cfg.min_gripper_delta
            )
            valid_rows.append((original_index, step, meaningful))

        if not valid_rows:
            return [], rejection_counts

        meaningful_positions = [pos for pos, (_, _, meaningful) in enumerate(valid_rows) if meaningful]
        if not meaningful_positions:
            rejection_counts["zero_or_tiny_action"] = rejection_counts.get("zero_or_tiny_action", 0) + len(valid_rows)
            return [], rejection_counts

        keep_positions: set[int] = set()
        for pos in meaningful_positions:
            start = max(0, pos - self.cfg.keep_idle_before_motion)
            end = min(len(valid_rows) - 1, pos + self.cfg.keep_idle_after_motion)
            keep_positions.update(range(start, end + 1))

        kept: list[dict[str, Any]] = []
        for pos, (_, step, meaningful) in enumerate(valid_rows):
            if pos in keep_positions:
                # Annotate for transparency; the RLDS converter can ignore this.
                step.setdefault("quality", {})["kept_by_cleaning"] = "meaningful_action" if meaningful else "idle_context"
                kept.append(step)
            else:
                reject("zero_or_tiny_action")

        return kept, rejection_counts

    def basic_rejection_reason(self, raw_dir: Path, step: dict[str, Any]) -> Optional[str]:
        """Return the reason a step fails basic quality checks, or None."""
        quality = step.get("quality", {})
        if self.cfg.require_camera_ok and quality.get("camera_ok") is not True:
            return "camera_not_ok"
        if self.cfg.require_tcp_available and quality.get("tcp_available") is not True:
            return "tcp_unavailable"
        if self.cfg.require_safety_ok and quality.get("safety_ok") is not True:
            return "safety_not_ok"

        image_path = nested_get(step, ["image", "path"])
        if self.cfg.require_image_file:
            if not image_path:
                return "missing_image_path"
            if not (raw_dir / image_path).exists():
                return "missing_image_file"

        pose = nested_get(step, ["follower", "tcp_pose_actual"])
        if not isinstance(pose, list) or len(pose) != 6:
            return "bad_tcp_pose"
        return None

    def reindex_and_recompute(self, raw_dir: Path, kept_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reindex retained observations and recompute actions between them."""
        cleaned: list[dict[str, Any]] = []
        for new_index, raw_step in enumerate(kept_steps):
            step = json.loads(json.dumps(raw_step))  # simple deep copy of JSON-safe data
            old_image_path = nested_get(step, ["image", "path"])
            step["step_index"] = new_index
            step["is_first"] = new_index == 0
            step["is_last"] = False
            step["is_terminal"] = False
            step["image"] = dict(step.get("image", {}))
            step["image"]["path"] = f"images/{new_index:06d}.jpg"
            step["image"]["saved"] = True
            step["_source_image_abs"] = str((raw_dir / old_image_path).resolve())
            step["action_to_next_raw"] = None
            cleaned.append(step)

        for i in range(len(cleaned) - 1):
            self.attach_action(cleaned[i], cleaned[i + 1])
        if cleaned:
            self.attach_terminal_action(cleaned[-1])
        return cleaned

    def attach_action(self, current: dict[str, Any], following: dict[str, Any]) -> None:
        """Attach an action from current retained step to following retained step."""
        pose_a = nested_get(current, ["follower", "tcp_pose_actual"])
        pose_b = nested_get(following, ["follower", "tcp_pose_actual"])
        delta = tcp_delta(pose_a, pose_b)
        grip_delta = gripper_delta(current, following)
        translation_m, rotation_rad = action_magnitudes(delta)
        dt_s = (following["host_monotonic_ns"] - current["host_monotonic_ns"]) / 1e9
        action_8d = None if grip_delta is None else delta + [grip_delta, 0.0]

        current["action_to_next_raw"] = {
            "tcp_delta_base_frame": delta,
            "gripper_delta_binary": grip_delta,
            "terminate": 0.0,
            "action_8d_raw": action_8d,
            "dt_s": dt_s,
            "recomputed_after_cleaning": True,
        }
        current.setdefault("quality", {}).update(
            {
                "action_valid": action_8d is not None,
                "translation_delta_m": translation_m,
                "rotation_delta_rad": rotation_rad,
                "small_motion": not has_meaningful_action(
                    delta, grip_delta, self.cfg.min_translation_m, self.min_rotation_rad, self.cfg.min_gripper_delta
                ),
            }
        )
        if action_8d is None:
            current["quality"]["action_invalid_reason"] = "missing_gripper_state"

    def attach_terminal_action(self, step: dict[str, Any]) -> None:
        """Mark the final retained step as terminal."""
        step["is_last"] = True
        step["is_terminal"] = True
        if not self.cfg.terminal_action:
            step["action_to_next_raw"] = None
            step.setdefault("quality", {}).update(
                {"action_valid": False, "action_invalid_reason": "final_step_has_no_next_state"}
            )
            return

        # The gripper value in this dataset is a DELTA action, not the absolute
        # final gripper state. A terminal action should therefore use 0.0 for
        # gripper_delta_binary and 1.0 only for the terminate dimension.
        grip_delta_zero = 0.0 if nested_get(step, ["gripper", "closed"]) is not None else None
        action_8d = None if grip_delta_zero is None else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, grip_delta_zero, 1.0]
        step["action_to_next_raw"] = {
            "tcp_delta_base_frame": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "gripper_delta_binary": grip_delta_zero,
            "terminate": 1.0,
            "action_8d_raw": action_8d,
            "dt_s": 0.0,
            "terminal_action": True,
        }
        step.setdefault("quality", {}).update(
            {
                "action_valid": action_8d is not None,
                "translation_delta_m": 0.0,
                "rotation_delta_rad": 0.0,
                "small_motion": False,
            }
        )
        if action_8d is None:
            step["quality"]["action_invalid_reason"] = "terminal_missing_gripper_state"

    @staticmethod
    def summary(
        episode_id: str,
        input_dir: Path,
        status: str,
        raw_steps: int,
        kept_steps: int,
        output_dir: Optional[Path] = None,
        rejection_counts: Optional[dict[str, int]] = None,
    ) -> dict[str, Any]:
        """Return one manifest summary row."""
        return {
            "episode_id": episode_id,
            "input_dir": str(input_dir.resolve()),
            "output_dir": None if output_dir is None else str(output_dir.resolve()),
            "status": status,
            "raw_steps": raw_steps,
            "kept_steps": kept_steps,
            "rejection_counts": rejection_counts or {},
        }


def parse_args() -> CleanConfig:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Clean raw UR5e/OpenVLA episode folders.")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-translation-m", type=float, default=0.002)
    parser.add_argument("--min-rotation-deg", type=float, default=1.0)
    parser.add_argument("--min-gripper-delta", type=float, default=0.5)
    parser.add_argument(
        "--keep-idle-before-motion",
        type=int,
        default=1,
        help="Number of adjacent idle/context steps to keep before each meaningful action.",
    )
    parser.add_argument(
        "--keep-idle-after-motion",
        type=int,
        default=1,
        help="Number of adjacent idle/context steps to keep after each meaningful action.",
    )
    parser.add_argument("--max-dt-s", type=float, default=0.6)
    parser.add_argument("--min-steps", type=int, default=5)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--allow-bad-camera", action="store_true")
    parser.add_argument("--allow-missing-tcp", action="store_true")
    parser.add_argument("--allow-bad-safety", action="store_true")
    parser.add_argument("--allow-missing-image-file", action="store_true")
    parser.add_argument("--no-terminal-action", action="store_true")
    args = parser.parse_args()

    if args.min_translation_m < 0 or args.min_rotation_deg < 0 or args.min_gripper_delta < 0:
        parser.error("motion/gripper thresholds must be non-negative")
    if args.keep_idle_before_motion < 0 or args.keep_idle_after_motion < 0:
        parser.error("idle-context counts must be non-negative")
    if args.max_dt_s <= 0:
        parser.error("--max-dt-s must be positive")
    if args.min_steps < 2:
        parser.error("--min-steps must be at least 2")

    return CleanConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        overwrite=args.overwrite,
        min_translation_m=args.min_translation_m,
        min_rotation_deg=args.min_rotation_deg,
        min_gripper_delta=args.min_gripper_delta,
        keep_idle_before_motion=args.keep_idle_before_motion,
        keep_idle_after_motion=args.keep_idle_after_motion,
        max_dt_s=args.max_dt_s,
        min_steps=args.min_steps,
        require_complete=not args.allow_incomplete,
        require_camera_ok=not args.allow_bad_camera,
        require_tcp_available=not args.allow_missing_tcp,
        require_safety_ok=not args.allow_bad_safety,
        require_image_file=not args.allow_missing_image_file,
        terminal_action=not args.no_terminal_action,
    )


def main() -> None:
    """Clean all episodes under the input root."""
    EpisodeCleaner(parse_args()).clean_all()


if __name__ == "__main__":
    main()
