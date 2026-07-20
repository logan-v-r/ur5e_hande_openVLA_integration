#!/usr/bin/env python3
"""
Convert a cleaned UR5e/OpenVLA dataset from gripper-delta actions to
absolute desired gripper-state actions.

The source dataset is not modified.

Source action convention
------------------------

    [dx, dy, dz, drx, dry, drz, gripper_delta, terminate]

where:

    +1.0 = open -> closed
    -1.0 = closed -> open
     0.0 = no gripper-state change

Destination action convention
-----------------------------

    [dx, dy, dz, drx, dry, drz, gripper_closed, terminate]

where:

    0.0 = desired gripper state is open
    1.0 = desired gripper state is closed

The absolute value is aligned with the action target. Therefore, a step
containing a +1 close delta receives an absolute action of 1.0 on that
same step. Following steps remain 1.0 until an open delta is encountered.

The script copies each complete episode to a separate output directory,
updates its steps.jsonl, and adds conversion metadata. Images and all
other episode files are preserved.

Example
-------

Run from the repository root:

    python data_processing/convert_gripper_delta_to_absolute.py \\
        --input-root ~/workspaces/openvla/datasets/ur5e_clean \\
        --output-root ~/workspaces/openvla/datasets/ur5e_clean_absolute

Use --overwrite to replace existing converted episodes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Write readable JSON."""
    path.write_text(
        json.dumps(
            value,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file."""
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc

    return rows


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write a JSON Lines file."""
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    allow_nan=False,
                )
                + "\n"
            )


def require_binary_state(
    value: Any,
    *,
    context: str,
) -> float:
    """
    Validate and return a binary gripper state.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context}: expected a numeric binary state, got {value!r}."
        ) from exc

    if numeric not in (0.0, 1.0):
        raise ValueError(
            f"{context}: expected 0.0 or 1.0, got {numeric}."
        )

    return numeric


def require_action_8d(
    step: dict[str, Any],
    *,
    episode_id: str,
    step_index: int,
) -> list[float]:
    """
    Read and validate the source eight-dimensional action.
    """
    action_record = step.get("action_to_next_raw")

    if not isinstance(action_record, dict):
        raise ValueError(
            f"{episode_id} step {step_index}: "
            "missing action_to_next_raw."
        )

    action = action_record.get("action_8d_raw")

    if not isinstance(action, list) or len(action) != 8:
        raise ValueError(
            f"{episode_id} step {step_index}: "
            "action_8d_raw must contain eight values."
        )

    try:
        numeric_action = [float(value) for value in action]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{episode_id} step {step_index}: "
            "action_8d_raw contains a non-numeric value."
        ) from exc

    return numeric_action


def infer_initial_state(
    steps: list[dict[str, Any]],
    *,
    episode_id: str,
    assume_open: bool,
) -> float:
    """
    Determine the gripper state at the first observation.

    Prefer the recorded gripper.closed value. Use open only as an explicit
    fallback when --assume-open is supplied.
    """
    if not steps:
        raise ValueError(f"{episode_id}: episode contains no steps.")

    recorded = (
        steps[0]
        .get("gripper", {})
        .get("closed")
    )

    if recorded is not None:
        return require_binary_state(
            recorded,
            context=f"{episode_id} first recorded gripper state",
        )

    if assume_open:
        return 0.0

    raise ValueError(
        f"{episode_id}: first step has no gripper.closed value. "
        "Use --assume-open only if every source episode truly starts open."
    )


def convert_episode_steps(
    steps: list[dict[str, Any]],
    *,
    episode_id: str,
    assume_open: bool,
    strict_state_check: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Replace source gripper deltas with absolute target states.
    """
    if not steps:
        raise ValueError(f"{episode_id}: no steps found.")

    current_state = infer_initial_state(
        steps,
        episode_id=episode_id,
        assume_open=assume_open,
    )

    counts: Counter[str] = Counter()
    mismatches: list[dict[str, Any]] = []

    converted_steps: list[dict[str, Any]] = []

    for step_index, source_step in enumerate(steps):
        # JSON-safe deep copy.
        step = json.loads(json.dumps(source_step))

        action = require_action_8d(
            step,
            episode_id=episode_id,
            step_index=step_index,
        )

        action_record = step["action_to_next_raw"]

        recorded_state_value = (
            step.get("gripper", {})
            .get("closed")
        )

        if recorded_state_value is not None:
            recorded_state = require_binary_state(
                recorded_state_value,
                context=(
                    f"{episode_id} step {step_index} "
                    "recorded gripper state"
                ),
            )

            if recorded_state != current_state:
                mismatch = {
                    "step_index": step_index,
                    "reconstructed_current_state": current_state,
                    "recorded_current_state": recorded_state,
                }

                mismatches.append(mismatch)

                if strict_state_check:
                    raise ValueError(
                        f"{episode_id} step {step_index}: "
                        f"reconstructed state {current_state} does not "
                        f"match recorded state {recorded_state}."
                    )

                # Recorded state is the most direct evidence for the
                # observation at this step.
                current_state = recorded_state

        gripper_delta = float(action[6])

        if gripper_delta >= 0.5:
            target_state = 1.0
            counts["close_transitions"] += 1
        elif gripper_delta <= -0.5:
            target_state = 0.0
            counts["open_transitions"] += 1
        else:
            target_state = current_state
            counts["maintain_state_actions"] += 1

        # Preserve arm deltas and terminal value. Replace only dimension 7.
        converted_action = (
            action[:6]
            + [target_state]
            + [action[7]]
        )

        # Preserve the original values for traceability.
        action_record[
            "source_gripper_delta_binary"
        ] = gripper_delta

        action_record[
            "source_action_8d_delta_gripper"
        ] = action.copy()

        # New explicit absolute-state fields.
        action_record[
            "gripper_absolute_binary"
        ] = target_state

        action_record[
            "gripper_action_representation"
        ] = "absolute_target_state"

        action_record[
            "action_8d_raw"
        ] = converted_action

        # Keep the old field out of the active representation so downstream
        # code does not accidentally treat the absolute value as a delta.
        action_record.pop(
            "gripper_delta_binary",
            None,
        )

        step.setdefault(
            "conversion",
            {},
        ).update(
            {
                "gripper_action_source": "delta",
                "gripper_action_destination": "absolute_target_state",
                "source_gripper_delta": gripper_delta,
                "absolute_target_state": target_state,
                "converted_utc": utc_now(),
            }
        )

        converted_steps.append(step)

        # The action target becomes the current state for the next
        # observation.
        current_state = target_state

        if target_state == 1.0:
            counts["closed_target_actions"] += 1
        else:
            counts["open_target_actions"] += 1

    summary = {
        "steps": len(converted_steps),
        "initial_gripper_state": infer_initial_state(
            steps,
            episode_id=episode_id,
            assume_open=assume_open,
        ),
        "final_gripper_state": current_state,
        "close_transitions": counts["close_transitions"],
        "open_transitions": counts["open_transitions"],
        "maintain_state_actions": counts["maintain_state_actions"],
        "closed_target_actions": counts["closed_target_actions"],
        "open_target_actions": counts["open_target_actions"],
        "recorded_state_mismatches": len(mismatches),
        "mismatch_examples": mismatches[:20],
    }

    return converted_steps, summary


def copy_episode(
    input_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    """
    Copy one complete episode directory.
    """
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}"
            )

        shutil.rmtree(output_dir)

    shutil.copytree(
        input_dir,
        output_dir,
    )


def convert_episode(
    input_dir: Path,
    output_root: Path,
    *,
    overwrite: bool,
    assume_open: bool,
    strict_state_check: bool,
) -> dict[str, Any]:
    """
    Copy and convert one cleaned episode.
    """
    episode_id = input_dir.name
    output_dir = output_root / episode_id

    required_paths = [
        input_dir / "steps.jsonl",
        input_dir / "episode_metadata.json",
        input_dir / "COMPLETE.json",
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"{episode_id}: missing required file {path.name}."
            )

    complete = read_json(
        input_dir / "COMPLETE.json"
    )

    if complete.get("cleaning_complete") is not True:
        raise ValueError(
            f"{episode_id}: source episode is not marked "
            "cleaning_complete."
        )

    source_steps = read_jsonl(
        input_dir / "steps.jsonl"
    )

    converted_steps, conversion_summary = (
        convert_episode_steps(
            source_steps,
            episode_id=episode_id,
            assume_open=assume_open,
            strict_state_check=strict_state_check,
        )
    )

    copy_episode(
        input_dir,
        output_dir,
        overwrite=overwrite,
    )

    write_jsonl(
        output_dir / "steps.jsonl",
        converted_steps,
    )

    metadata_path = (
        output_dir / "episode_metadata.json"
    )

    metadata = read_json(metadata_path)

    metadata[
        "gripper_action_conversion"
    ] = {
        "converted_utc": utc_now(),
        "source_episode_dir": str(
            input_dir.resolve()
        ),
        "representation": "absolute_target_state",
        "action_format": [
            "dx",
            "dy",
            "dz",
            "drx",
            "dry",
            "drz",
            "gripper_closed",
            "terminate",
        ],
        **conversion_summary,
    }

    write_json(
        metadata_path,
        metadata,
    )

    complete_path = output_dir / "COMPLETE.json"
    output_complete = read_json(complete_path)

    output_complete.update(
        {
            "gripper_absolute_conversion_complete": True,
            "gripper_absolute_conversion_utc": utc_now(),
            "source_episode_dir": str(
                input_dir.resolve()
            ),
            "gripper_action_representation": (
                "absolute_target_state"
            ),
        }
    )

    write_json(
        complete_path,
        output_complete,
    )

    return {
        "episode_id": episode_id,
        "status": "converted",
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        **conversion_summary,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Copy a cleaned UR5e/OpenVLA dataset and convert "
            "gripper-delta actions to absolute target states."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help=(
            "Source cleaned dataset containing episode_* directories."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Destination for the converted dataset. It must differ "
            "from --input-root."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace converted episode directories that already exist.",
    )

    parser.add_argument(
        "--assume-open",
        action="store_true",
        help=(
            "Assume the first state is open only when the first step "
            "does not contain gripper.closed."
        ),
    )

    parser.add_argument(
        "--strict-state-check",
        action="store_true",
        help=(
            "Fail if the state reconstructed from deltas disagrees "
            "with a step's recorded gripper.closed state."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Convert every cleaned episode."""
    args = parse_args()

    input_root = (
        args.input_root
        .expanduser()
        .resolve()
    )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    if not input_root.exists():
        raise FileNotFoundError(
            f"Input root does not exist: {input_root}"
        )

    if input_root == output_root:
        raise ValueError(
            "Input and output roots must be different."
        )

    episode_dirs = sorted(
        path
        for path in input_root.glob("episode_*")
        if path.is_dir()
    )

    if not episode_dirs:
        raise FileNotFoundError(
            f"No episode_* directories found under {input_root}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest: list[dict[str, Any]] = []

    for episode_dir in episode_dirs:
        try:
            result = convert_episode(
                episode_dir,
                output_root,
                overwrite=args.overwrite,
                assume_open=args.assume_open,
                strict_state_check=args.strict_state_check,
            )
        except FileExistsError as exc:
            result = {
                "episode_id": episode_dir.name,
                "status": "skipped_output_exists",
                "error": str(exc),
            }
        except Exception as exc:
            result = {
                "episode_id": episode_dir.name,
                "status": "error",
                "error": str(exc),
            }

        manifest.append(result)

        print(
            f"{result['episode_id']}: "
            f"{result['status']}"
        )

        if result["status"] == "converted":
            print(
                "  close transitions: "
                f"{result['close_transitions']}"
            )
            print(
                "  open transitions: "
                f"{result['open_transitions']}"
            )
            print(
                "  closed target actions: "
                f"{result['closed_target_actions']}"
            )
            print(
                "  open target actions: "
                f"{result['open_target_actions']}"
            )
            print(
                "  state mismatches: "
                f"{result['recorded_state_mismatches']}"
            )
        elif "error" in result:
            print(
                f"  {result['error']}"
            )

    manifest_path = (
        output_root
        / "gripper_absolute_conversion_manifest.jsonl"
    )

    write_jsonl(
        manifest_path,
        manifest,
    )

    converted_count = sum(
        row.get("status") == "converted"
        for row in manifest
    )

    error_count = sum(
        row.get("status") == "error"
        for row in manifest
    )

    print("\nConversion complete.")
    print(f"Converted episodes: {converted_count}")
    print(f"Errors: {error_count}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
