'''
import numpy as np

MAX_STEP_METERS = 0.01  # Start with 1 cm per axis.


def openvla_to_ur5_target(current_tcp_pose, openvla_action):
    """
    Convert OpenVLA XYZ output into a conservative UR5 TCP target.

    Uses translation only.
    Keeps the current UR5 orientation unchanged.
    """

    world_vector = np.asarray(
        openvla_action["world_vector"],
        dtype=float,
    )

    if world_vector.shape != (3,):
        raise ValueError("world_vector must contain [dx, dy, dz].")

    # Preserve OpenVLA's small physical delta values,
    # but hard-cap each axis for the real robot.
    safe_delta = np.clip(
        world_vector,
        -MAX_STEP_METERS,
        MAX_STEP_METERS,
    )

    target_tcp = list(current_tcp_pose)

    target_tcp[0] += float(safe_delta[0])
    target_tcp[1] += float(safe_delta[1])
    target_tcp[2] += float(safe_delta[2])

    # Leave [rx, ry, rz] unchanged for early tests.
    return target_tcp
'''

from __future__ import annotations

import robotiq_gripper
import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Conservative early-stage limits
# ---------------------------------------------------------------------------

# Maximum movement per Cartesian axis for one OpenVLA action.
MAX_TRANSLATION_METERS = 0.0254  # 2.54 cm = 1 in

# Maximum total orientation change for one OpenVLA action.
# 0.1 radians is about 5.7 degrees.
MAX_ROTATION_RADIANS = 3.1415  # 90 degrees

# Keep False until translation-only tests are predictable and safe.
USE_ROTATION = True

# OpenVLA rotation deltas are interpreted as tool-relative for this first
# implementation. See notes below before changing this.
ROTATION_FRAME = "tool"

# Keep False until the gripper has been tested independently.
USE_GRIPPER = True

# The OpenVLA wrapper is expected to provide a scalar gripper signal.
# This mapping assumes:
#   >= 0 means open
#   < 0 means close
#
# Verify this with a one-command physical test before enabling USE_GRIPPER.
GRIPPER_OPEN_THRESHOLD = 0.0


def openvla_to_ur5_target(
    current_tcp_pose: list[float],
    openvla_action: dict[str, np.ndarray],
) -> list[float]:
    """
    Convert an OpenVLA action into a conservative UR5 TCP target pose.

    Inputs:
        current_tcp_pose:
            [x, y, z, rx, ry, rz] from rtde_r.getActualTCPPose()

        openvla_action:
            {
                "world_vector": [dx, dy, dz],
                "rot_axangle": [drx, dry, drz],
                ...
            }

    Behavior:
        - clamps XYZ translation;
        - optionally clamps and composes rotation;
        - leaves orientation unchanged while USE_ROTATION is False;
        - does not command the gripper.
    """

    current_tcp = np.asarray(current_tcp_pose, dtype=float)

    if current_tcp.shape != (6,):
        raise ValueError(
            "current_tcp_pose must contain [x, y, z, rx, ry, rz]."
        )

    world_vector = np.asarray(
        openvla_action["world_vector"],
        dtype=float,
    )

    rot_axangle = np.asarray(
        openvla_action["rot_axangle"],
        dtype=float,
    )

    if world_vector.shape != (3,):
        raise ValueError(
            "OpenVLA world_vector must contain exactly [dx, dy, dz]."
        )

    if rot_axangle.shape != (3,):
        raise ValueError(
            "OpenVLA rot_axangle must contain exactly [drx, dry, drz]."
        )

    # -----------------------------------------------------------------------
    # Translation
    # -----------------------------------------------------------------------

    safe_translation = np.clip(
        world_vector,
        -MAX_TRANSLATION_METERS,
        MAX_TRANSLATION_METERS,
    )

    target_tcp = current_tcp.copy()
    target_tcp[:3] += safe_translation

    # -----------------------------------------------------------------------
    # Rotation
    # -----------------------------------------------------------------------

    if not USE_ROTATION:
        # Preserve the current UR5 tool orientation.
        return target_tcp.tolist()

    
    safe_rotation = np.clip(
        rot_axangle,
        -MAX_ROTATION_RADIANS,
        MAX_ROTATION_RADIANS,
    )

    ROTATION_AXIS_MASK = np.array([1.0, 1.0, 1.0])
    safe_rotation *= ROTATION_AXIS_MASK
    
    target_tcp[3:] += safe_rotation
    
    '''
    # Limit the TOTAL magnitude of the rotation, not each component
    # independently. This preserves the intended rotation axis.
    rotation_magnitude = np.linalg.norm(rot_axangle)

    if rotation_magnitude > MAX_ROTATION_RADIANS:
        safe_rot_delta = (
            rot_axangle / rotation_magnitude
        ) * MAX_ROTATION_RADIANS
    else:
        safe_rot_delta = rot_axangle
    
    # Optional: temporarily suppress a problematic model rotation axis.
    # Start with all axes enabled. Set an axis to 0.0 only after testing.
    ROTATION_AXIS_MASK = np.array([1.0, 1.0, 1.0])

    safe_rot_delta *= ROTATION_AXIS_MASK

    print("Raw OpenVLA rotation:", rot_axangle)
    print("Clamped OpenVLA rotation:", safe_rot_delta)
    print("Rotation frame:", ROTATION_FRAME)

    current_rotation = Rotation.from_rotvec(current_tcp[3:])
    delta_rotation = Rotation.from_rotvec(safe_rot_delta)

    if ROTATION_FRAME == "tool":
        # Apply OpenVLA's rotation in the current tool frame.
        target_rotation = current_rotation * delta_rotation

    elif ROTATION_FRAME == "base":
        # Apply OpenVLA's rotation in the UR base frame.
        target_rotation = delta_rotation * current_rotation

    else:
        raise ValueError(
            "ROTATION_FRAME must be either 'tool' or 'base'."
        )

    target_tcp[3:] = target_rotation.as_rotvec()
    '''
    
    return target_tcp.tolist()

def openvla_to_gripper_command(
    openvla_action: dict,
) -> str | None:
    """
    Translate OpenVLA's processed gripper output into a high-level command.

    Returns:
        "open"
        "close"
        None, when gripper control is disabled.

    This function deliberately does not connect to or command hardware.
    """

    if not USE_GRIPPER:
        return None

    if "gripper" not in openvla_action:
        raise KeyError(
            "OpenVLA action does not contain a 'gripper' value."
        )

    gripper_value = float(
        np.asarray(openvla_action["gripper"]).reshape(-1)[0]
    )

    if gripper_value >= GRIPPER_OPEN_THRESHOLD:
        return "open"

    return "close"