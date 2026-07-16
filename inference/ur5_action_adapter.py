'''
Convert processed OpenVLA actions into UR5e target commands.

This module sits between model inference and physical robot execution.
It does not connect to or command the UR5e or Robotiq Hand-E gripper.

For the custom fine-tuned `ur5e_openvla` model, the processed action is
expected to contain:

    world_vector:
        Relative Cartesian translation [dx, dy, dz] expressed in the
        UR5e base frame, in metres.

    rot_axangle:
        Relative orientation change [drx, dry, drz] expressed as a
        base-frame rotation vector. The vector direction defines the
        rotation axis and its magnitude defines the angle in radians.

    gripper:
        Processed scalar gripper command.

The translation and rotation conventions must remain consistent with
`clean_raw_episodes.py` and the custom RLDS dataset builder.

The target orientation is reconstructed as:

    target_rotation = delta_rotation * current_rotation

This is the inverse of the dataset-cleaning calculation:

    delta_rotation = target_rotation * inverse(current_rotation)

The adapter limits each predicted action before returning:

    - a six-element UR5e TCP target pose; and
    - a high-level gripper command: "open", "close", or None.

Workspace checks, inverse kinematics, hardware communication, and motion
execution are handled by `openvla_move_with_liveview.py`.
'''

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Per-action limits and feature controls
# ---------------------------------------------------------------------------

# Maximum absolute translation allowed along each base-frame Cartesian
# axis for one predicted action. Units are metres.
MAX_TRANSLATION_METERS = 0.0254  # 2.54 cm, or 1 inch, per axis.

# Maximum magnitude of the complete rotation delta for one predicted
# action. This limits the norm of the rotation vector, not each component
# independently. Units are radians.
MAX_ROTATION_RADIANS = 0.1  # Approximately 5.7 degrees.

# When False, preserve the current UR5e orientation and ignore the
# predicted rotation delta.
USE_ROTATION = True

# Optional component mask applied to the base-frame rotation vector.
# Keep all axes enabled for normal operation.
ROTATION_AXIS_MASK = np.array([1.0, 1.0, 1.0], dtype=np.float64)

# When False, openvla_to_gripper_command() returns None.
USE_GRIPPER = True

# Processed gripper convention currently used by the inference wrapper:
#   value >= 0.0 -> open
#   value <  0.0 -> close
GRIPPER_OPEN_THRESHOLD = 0.0

# ---------------------------------------------------------------------------
# Rotation helper functions
# ---------------------------------------------------------------------------

def _require_vector(
    action: Mapping[str, Any],
    key: str,
    expected_size: int,
) -> np.ndarray:
    """
    Read and validate a finite one-dimensional action vector.
    """
    if key not in action:
        raise KeyError(f"OpenVLA action does not contain {key!r}.")

    value = np.asarray(action[key], dtype=np.float64).reshape(-1)

    if value.size != expected_size:
        raise ValueError(
            f"OpenVLA action {key!r} must contain {expected_size} values, "
            f"got shape {np.asarray(action[key]).shape}."
        )

    if not np.all(np.isfinite(value)):
        raise ValueError(
            f"OpenVLA action {key!r} contains non-finite values: {value}."
        )

    return value

def _limit_vector_norm(
    vector: np.ndarray,
    max_norm: float,
) -> np.ndarray:
    """
    Return a copy of a vector whose magnitude does not exceed max_norm.
    """
    if max_norm < 0:
        raise ValueError("max_norm must be non-negative.")

    norm = float(np.linalg.norm(vector))

    if norm == 0.0 or norm <= max_norm:
        return vector.copy()

    return vector * (max_norm / norm)

# ---------------------------------------------------------------------------
# Convert Actions
# ---------------------------------------------------------------------------

def openvla_to_ur5_target(
    current_tcp_pose: Sequence[float],
    openvla_action: Mapping[str, Any],
) -> list[float]:
    """
    Convert one processed fine-tuned OpenVLA action into a UR5e TCP target.

    Args:
        current_tcp_pose:
            Current UR5e TCP pose from `getActualTCPPose()`, represented
            as [x, y, z, rx, ry, rz]. Position is expressed in metres.
            Orientation is a UR rotation vector expressed in radians.

        openvla_action:
            Processed action containing:
                world_vector:
                    Base-frame relative XYZ translation in metres.
                rot_axangle:
                    Base-frame relative rotation vector in radians.

    Returns:
        A six-element target TCP pose:
            [x, y, z, rx, ry, rz]

    Notes:
        - Translation is limited per Cartesian axis.
        - Rotation is limited by the total rotation-vector magnitude.
        - The rotation delta is left-composed because the fine-tuning
          dataset stores base-frame rotation deltas:

              target = delta * current

        - This function does not perform workspace, safety, or inverse-
          kinematics checks and does not command the robot.
    """
    current_tcp = np.asarray(current_tcp_pose, dtype=np.float64).reshape(-1)

    if current_tcp.size != 6:
        raise ValueError(
            "current_tcp_pose must contain six values "
            "[x, y, z, rx, ry, rz], "
            f"got {current_tcp.size}."
        )

    if not np.all(np.isfinite(current_tcp)):
        raise ValueError(
            f"current_tcp_pose contains non-finite values: {current_tcp}."
        )

    translation_delta = _require_vector(
        openvla_action,
        "world_vector",
        3,
    )

    safe_translation = np.clip(
        translation_delta,
        -MAX_TRANSLATION_METERS,
        MAX_TRANSLATION_METERS,
    )

    target_tcp = current_tcp.copy()
    target_tcp[:3] = current_tcp[:3] + safe_translation

    if not USE_ROTATION:
        return target_tcp.tolist()

    rotation_delta = _require_vector(
        openvla_action,
        "rot_axangle",
        3,
    )

    masked_rotation_delta = rotation_delta * ROTATION_AXIS_MASK

    safe_rotation_delta = _limit_vector_norm(
        masked_rotation_delta,
        MAX_ROTATION_RADIANS,
    )

    current_rotation = Rotation.from_rotvec(current_tcp[3:6])
    delta_rotation = Rotation.from_rotvec(safe_rotation_delta)

    # The custom dataset stores:
    #
    #     delta = next * inverse(current)
    #
    # Therefore reconstruct the target with:
    #
    #     next = delta * current
    target_rotation = delta_rotation * current_rotation

    target_tcp[3:6] = target_rotation.as_rotvec()

    return target_tcp.tolist()

def openvla_to_gripper_command(
    openvla_action: Mapping[str, Any],
) -> str | None:
    """
    Map OpenVLA's processed gripper output to a high-level command.

    Returns:
        "open":
            When gripper control is enabled and the processed value is
            greater than or equal to GRIPPER_OPEN_THRESHOLD.

        "close":
            When gripper control is enabled and the processed value is
            below GRIPPER_OPEN_THRESHOLD.

        None:
            When gripper control is disabled.

    This function only interprets the model output. Gripper communication
    and physical execution are handled by
    `openvla_move_with_liveview.py`.
    """
    if not USE_GRIPPER:
        return None

    gripper_value = _require_vector(
        openvla_action,
        "gripper",
        1,
    )[0]

    if gripper_value >= GRIPPER_OPEN_THRESHOLD:
        return "open"

    return "close"

# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------

def describe_action_limits(
    original: np.ndarray,
    limited: np.ndarray,
) -> str:
    """Return a concise description when an action was limited."""
    return (
        f"raw={np.array2string(original, precision=5)}, "
        f"limited={np.array2string(limited, precision=5)}"
    )
