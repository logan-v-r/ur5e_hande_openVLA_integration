from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation

import ur5_action_adapter as action_adapter
from openvla_inference import OpenVLAInference
from ur5_action_adapter import (
    openvla_to_ur5_target,
    openvla_to_gripper_command,
)

import robotiq_gripper


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROBOT_IP = "192.168.1.102"

# Set this to the RealSense stream that gave you a normal full-color view.
# Your old camera index 4 may no longer be correct after reconnecting it.
CAMERA_INDEX = 4

# Start with one physical movement per run.
MAX_STEPS = 15

# Pause only matters if MAX_STEPS is increased above 1.
INTER_STEP_PAUSE_SECONDS = 0.1

# Conservative early physical-test settings.
SPEED_M_PER_S = 0.1
ACCEL_M_PER_S2 = 0.8

# Stops a current asynchronous move if Q or Esc is pressed.
STOP_ACCEL_M_PER_S2 = 0.5

# Maximum time allowed for one movement.
MOVE_TIMEOUT_SECONDS = 10.0

# Target-pose tolerance used to determine that the asynchronous move is done.
POSITION_TOLERANCE_METERS = 0.0005
ORIENTATION_TOLERANCE_RADIANS = 0.005

USE_GRIPPER = True

GRIPPER_IP = "192.168.1.102"
GRIPPER_PORT = 63352

# Conservative values for initial physical tests.
GRIPPER_SPEED = 64
GRIPPER_FORCE = 20

GRIPPER_OPEN_POSITION = 0
GRIPPER_CLOSED_POSITION = 255

# Keep this simple during early testing.
INSTRUCTION = "pick up the red block"

OUTPUT_DIR = os.path.expanduser(
    "~/workspaces/openvla/ur5_rtde/logs/translation_rotation_and_gripper/test3_trial4"
)

SHOW_LIVE_PREVIEW = True
PREVIEW_WINDOW_NAME = "OpenVLA UR5 Live Feed"


def save_frame(bgr_frame: np.ndarray, step_number: int) -> str:
    """Save the frame used for one OpenVLA inference step."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    image_path = os.path.join(
        OUTPUT_DIR,
        f"move_test_step_{step_number}_{timestamp}.jpg",
    )

    if not cv2.imwrite(image_path, bgr_frame):
        raise RuntimeError(f"Could not save camera image to {image_path}.")

    return image_path


def get_camera_frame(capture) -> tuple[np.ndarray, np.ndarray]:
    """
    Read one camera frame.

    Returns:
        bgr_frame: OpenCV BGR image, used for display and logging.
        rgb_frame: RGB image, sent to OpenVLA.
    """

    ok, bgr_frame = capture.read()

    if not ok or bgr_frame is None:
        raise RuntimeError("Could not read a frame from the RealSense camera.")

    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    return bgr_frame, rgb_frame


def rotation_error_radians(
    actual_tcp: np.ndarray,
    target_tcp: np.ndarray,
) -> float:
    """
    Calculate orientation error properly using rotation composition.

    Do not subtract axis-angle vectors directly because equivalent rotations
    can have different axis-angle representations.
    """

    actual_rotation = Rotation.from_rotvec(actual_tcp[3:])
    target_rotation = Rotation.from_rotvec(target_tcp[3:])

    relative_rotation = target_rotation.inv() * actual_rotation

    return relative_rotation.magnitude()


def show_preview_frame(
    bgr_frame: np.ndarray,
    status_text: str,
    elapsed_seconds: float | None = None,
) -> int:
    """
    Display one live preview frame.

    Returns:
        The pressed key code, or -1 if no key was pressed.
    """

    preview_frame = bgr_frame.copy()

    cv2.putText(
        preview_frame,
        status_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    if elapsed_seconds is not None:
        cv2.putText(
            preview_frame,
            f"Move time: {elapsed_seconds:.1f}s",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    cv2.imshow(PREVIEW_WINDOW_NAME, preview_frame)

    return cv2.waitKey(1) & 0xFF

def connect_gripper() -> robotiq_gripper.RobotiqGripper:
    """
    Connect to and activate the Robotiq Hand-E once per script execution.

    auto_calibrate=False is intentional for early testing:
    the supplied driver's full auto-calibration routine opens, closes,
    and opens the gripper automatically.
    """

    print(f"Connecting to Robotiq gripper at {GRIPPER_IP}:{GRIPPER_PORT}...")

    gripper = robotiq_gripper.RobotiqGripper()

    gripper.connect(
        GRIPPER_IP,
        GRIPPER_PORT,
    )

    gripper.activate(auto_calibrate=False)

    print("Robotiq gripper is active.")

    return gripper


def execute_gripper_command(
    gripper: robotiq_gripper.RobotiqGripper,
    command: str | None,
) -> None:
    """
    Execute one high-level open/close command.

    Runs after the UR5 has reached its TCP target.
    """

    if command is None:
        return

    if command == "open":
        position = GRIPPER_OPEN_POSITION

    elif command == "close":
        position = GRIPPER_CLOSED_POSITION

    else:
        raise ValueError(
            f"Unsupported gripper command: {command}"
        )

    print(
        f"Executing gripper command: {command} "
        f"(position={position})"
    )

    final_position, object_status = gripper.move_and_wait_for_pos(
        position,
        GRIPPER_SPEED,
        GRIPPER_FORCE,
    )

    print(
        "Gripper result:",
        f"position={final_position},",
        f"status={object_status.name}",
    )


def execute_move_with_live_preview(
    rtde_c,
    rtde_r,
    capture,
    target_tcp: list[float],
) -> None:
    """
    Execute one asynchronous moveL() while updating the live camera feed.

    Press Q or Esc while the preview window is focused to stop the robot.
    """

    target_tcp_array = np.asarray(target_tcp, dtype=float)

    print("\nStarting asynchronous moveL() with live camera preview...")

    rtde_c.moveL(
        target_tcp,
        SPEED_M_PER_S,
        ACCEL_M_PER_S2,
        True,
    )

    move_start_time = time.monotonic()

    while True:
        ok, bgr_frame = capture.read()

        if not ok or bgr_frame is None:
            rtde_c.stopL(STOP_ACCEL_M_PER_S2)

            raise RuntimeError(
                "Could not read a camera frame while the robot was moving."
            )

        elapsed_seconds = time.monotonic() - move_start_time

        if SHOW_LIVE_PREVIEW:
            key = show_preview_frame(
                bgr_frame,
                "ROBOT MOVING - Press Q or Esc to stop",
                elapsed_seconds,
            )

            if key in (ord("q"), 27):
                print("\nStop requested from the preview window.")

                rtde_c.stopL(STOP_ACCEL_M_PER_S2)

                raise KeyboardInterrupt(
                    "Motion stopped from live preview window."
                )

        actual_tcp = np.asarray(
            rtde_r.getActualTCPPose(),
            dtype=float,
        )

        position_error = np.linalg.norm(
            actual_tcp[:3] - target_tcp_array[:3]
        )

        orientation_error = rotation_error_radians(
            actual_tcp,
            target_tcp_array,
        )

        if (
            position_error <= POSITION_TOLERANCE_METERS
            and orientation_error <= ORIENTATION_TOLERANCE_RADIANS
        ):
            print("Robot reached the target pose.")
            return

        if elapsed_seconds > MOVE_TIMEOUT_SECONDS:
            rtde_c.stopL(STOP_ACCEL_M_PER_S2)

            raise TimeoutError(
                "Robot did not reach the target before "
                "MOVE_TIMEOUT_SECONDS."
            )

        time.sleep(0.01)


def main() -> None:
    """
    For each configured step:

    1. Capture one fresh RealSense image.
    2. Run OpenVLA inference.
    3. Convert the action through ur5_action_adapter.py.
    4. Check UR safety limits and inverse kinematics.
    5. Execute one asynchronous moveL() with a live camera preview.
    """

    if MAX_STEPS < 1:
        raise ValueError("MAX_STEPS must be at least 1.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    capture = None
    rtde_c = None
    gripper = None
    last_gripper_command = None

    try:
        print(f"Connecting to UR5 receive interface at {ROBOT_IP}...")
        rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

        print(f"Connecting to UR5 control interface at {ROBOT_IP}...")
        rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
        
        if USE_GRIPPER:
            gripper = connect_gripper()

        print(f"Opening RealSense camera index {CAMERA_INDEX}...")
        capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open RealSense camera index {CAMERA_INDEX}."
            )

        if SHOW_LIVE_PREVIEW:
            cv2.namedWindow(
                PREVIEW_WINDOW_NAME,
                cv2.WINDOW_NORMAL,
            )

        # Allow exposure and white balance to stabilize.
        for _ in range(60):
            ok, _ = capture.read()

            if not ok:
                raise RuntimeError(
                    "Could not read a camera frame during warm-up."
                )

        print("\nLoading OpenVLA...")

        policy = OpenVLAInference(
            saved_model_path="openvla/openvla-7b",
            policy_setup="google_robot",
        )

        policy.reset(INSTRUCTION)

        print("\n" + "=" * 70)
        print("OPENVLA → UR5 MOVE TEST WITH LIVE CAMERA VIEW")
        print("=" * 70)
        print(f"Instruction: {INSTRUCTION}")
        print(f"Camera index: {CAMERA_INDEX}")
        print(f"Maximum steps: {MAX_STEPS}")
        print(
            "Rotation enabled in adapter:",
            getattr(action_adapter, "USE_ROTATION", "unknown"),
        )
        print(
            "Rotation frame in adapter:",
            getattr(action_adapter, "ROTATION_FRAME", "unknown"),
        )
        print("Gripper: disabled")
        print("Live preview: enabled")
        print("Press Q or Esc in the preview window to stop a move.")

        for step_number in range(1, MAX_STEPS + 1):
            print("\n" + "-" * 70)
            print(f"STEP {step_number} OF {MAX_STEPS}")
            print("-" * 70)

            # Capture the image used for this OpenVLA inference.
            bgr_frame, rgb_frame = get_camera_frame(capture)

            if SHOW_LIVE_PREVIEW:
                show_preview_frame(
                    bgr_frame,
                    "OPENVLA INFERENCE - Robot stationary",
                )

            image_path = save_frame(bgr_frame, step_number)

            raw_action, action = policy.step(
                rgb_frame,
                INSTRUCTION,
            )

            # Read current TCP immediately before generating the target.
            current_tcp = list(rtde_r.getActualTCPPose())

            # The adapter handles translation clamping and optional rotation.
            target_tcp = openvla_to_ur5_target(
                current_tcp,
                action,
            )
            
            gripper_command = openvla_to_gripper_command(action)

            print("\nOpenVLA gripper value:")
            print(action["gripper"])

            print("\nMapped gripper command:")
            print(gripper_command)

            pose_is_safe = rtde_c.isPoseWithinSafetyLimits(target_tcp)

            ik_exists = rtde_c.getInverseKinematicsHasSolution(target_tcp)

            print("\nSaved camera image:")
            print(image_path)

            print("\nCurrent TCP pose [x, y, z, rx, ry, rz]:")
            print(current_tcp)

            print("\nRaw OpenVLA action:")
            print(raw_action)

            print("\nProcessed OpenVLA action:")
            print(action)

            print("\nOpenVLA translation proposal [dx, dy, dz]:")
            print(action["world_vector"])

            print("\nOpenVLA rotation proposal [drx, dry, drz]:")
            print(action["rot_axangle"])

            print("\nProposed UR5 target TCP pose:")
            print(target_tcp)

            print("\nWithin configured UR safety limits:")
            print(pose_is_safe)

            print("\nInverse-kinematics solution exists:")
            print(ik_exists)

            if not pose_is_safe:
                raise RuntimeError(
                    "Target pose was rejected by configured UR safety limits."
                )

            if not ik_exists:
                raise RuntimeError(
                    "Target pose does not have a valid "
                    "inverse-kinematics solution."
                )

            execute_move_with_live_preview(
                rtde_c,
                rtde_r,
                capture,
                target_tcp,
            )
            
            if (
                gripper is not None
                and gripper_command is not None
                and gripper_command != last_gripper_command
            ):
                execute_gripper_command(
                    gripper,
                    gripper_command,
                )

                last_gripper_command = gripper_command

            print("\nTCP pose after move:")
            print(rtde_r.getActualTCPPose())

            if step_number < MAX_STEPS:
                print(
                    f"\nWaiting {INTER_STEP_PAUSE_SECONDS:.1f} seconds "
                    "before the next image and inference step..."
                )

                time.sleep(INTER_STEP_PAUSE_SECONDS)

        print("\nCompleted all configured OpenVLA motion steps.")

    except KeyboardInterrupt as exc:
        print(f"\nStopped: {exc}")

        if rtde_c is not None:
            print("Sending stopL()...")
            rtde_c.stopL(STOP_ACCEL_M_PER_S2)

    except Exception as exc:
        print(f"\nMove test failed: {exc}")

        if rtde_c is not None:
            print("Sending stopL()...")
            rtde_c.stopL(STOP_ACCEL_M_PER_S2)

        sys.exit(1)

    finally:
        if capture is not None:
            capture.release()

        cv2.destroyAllWindows()

        if rtde_c is not None:
            rtde_c.stopScript()
            
        if gripper is not None:
            gripper.disconnect()

        print("\nConnections closed.")


if __name__ == "__main__":
    main()