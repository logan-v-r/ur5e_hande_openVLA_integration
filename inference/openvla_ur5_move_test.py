from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import cv2
import rtde_control
import rtde_receive

from openvla_inference import OpenVLAInference
from ur5_action_adapter import openvla_to_ur5_target


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROBOT_IP = "192.168.1.102"

# RealSense D435i RGB stream.
CAMERA_INDEX = 4

# Start with 1. Increase only after reviewing successful, safe behavior.
MAX_STEPS = 40

# Require an explicit terminal confirmation before every movement.
REQUIRE_CONFIRMATION = False

# Pause after a completed move before capturing the next image.
INTER_STEP_PAUSE_SECONDS = 1.0

# Keep these deliberately slow for early supervised tests.
SPEED_M_PER_S = 0.1
ACCEL_M_PER_S2 = 0.5

# Do not use a grasp instruction for the first movement tests.
# First verify whether OpenVLA output changes meaningfully with object position.
INSTRUCTION = "pick up the red block"

# Display-only reminder. Must match ur5_action_adapter.py.
ROTATION_FRAME = "base"

OUTPUT_DIR = os.path.expanduser(
    "~/workspaces/openvla/ur5_rtde/logs"
)


def save_frame(bgr_frame, step_number: int) -> str:
    """Save the camera image used for one OpenVLA inference step."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(
        OUTPUT_DIR,
        f"move_test_step_{step_number}_{timestamp}.jpg",
    )

    if not cv2.imwrite(image_path, bgr_frame):
        raise RuntimeError(f"Could not save camera image to {image_path}.")

    return image_path


def get_camera_frame(capture):
    """
    Read one RealSense frame and convert BGR to RGB for OpenVLA.

    Returns:
        bgr_frame: Original OpenCV frame, used for logging.
        rgb_frame: uint8 RGB NumPy image, sent to OpenVLA.
    """
    ok, bgr_frame = capture.read()

    if not ok or bgr_frame is None:
        raise RuntimeError("Could not read a frame from the RealSense camera.")

    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    return bgr_frame, rgb_frame


def confirm_move(step_number: int) -> None:
    """
    Require an explicit operator confirmation before commanding motion.

    Type MOVE to proceed. Any other response exits without moving.
    """
    response = input(
        f"\nStep {step_number}: type MOVE to execute this one motion "
        "or press Enter to stop: "
    ).strip()

    if response != "MOVE":
        raise KeyboardInterrupt("Motion cancelled by operator.")


def main() -> None:
    """
    Run a small number of supervised OpenVLA-to-UR5 motion steps.

    Per step:
      1. Capture one fresh RGB camera image.
      2. Run OpenVLA inference.
      3. Read the current TCP pose.
      4. Create a bounded translation-only target TCP pose.
      5. Check UR configured safety limits.
      6. Print all proposed values.
      7. Require operator confirmation.
      8. Execute one slow, blocking moveL().
      9. Pause, then continue to the next step if configured.

    This version does not:
      - control a gripper;
      - run continuously;
      - use servoL();
      - move unless the operator types MOVE.
    """

    if MAX_STEPS < 1:
        raise ValueError("MAX_STEPS must be at least 1.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    capture = None
    rtde_c = None

    try:
        print(f"Connecting to UR5 receive interface at {ROBOT_IP}...")
        rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

        print(f"Connecting to UR5 control interface at {ROBOT_IP}...")
        rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)

        print(f"Opening RealSense RGB camera index {CAMERA_INDEX}...")
        capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open RealSense camera index {CAMERA_INDEX}."
            )

        # Allow camera auto-exposure and white balance to stabilize.
        for _ in range(60):
            ok, _ = capture.read()
            if not ok:
                raise RuntimeError("Could not read a camera frame during warm-up.")

        print("\nLoading OpenVLA...")
        policy = OpenVLAInference(
            saved_model_path="openvla/openvla-7b",
            policy_setup="google_robot",
        )
        policy.reset(INSTRUCTION)

        print("\n" + "=" * 70)
        print("SUPERVISED OPENVLA → UR5 TEST")
        print("=" * 70)
        print(f"Instruction: {INSTRUCTION}")
        print(f"Maximum steps this run: {MAX_STEPS}")
        print("Rotation: enabled through ur5_action_adapter.py")
        print(f"Rotation frame: {ROTATION_FRAME}")
        print("Gripper: disabled")
        print(f"Motion confirmation: {REQUIRE_CONFIRMATION}")

        for step_number in range(1, MAX_STEPS + 1):
            print("\n" + "-" * 70)
            print(f"STEP {step_number} OF {MAX_STEPS}")
            print("-" * 70)

            # Capture a fresh image before every model inference.
            bgr_frame, rgb_frame = get_camera_frame(capture)

            image_path = save_frame(bgr_frame, step_number)

            # Infer one OpenVLA action from the current image.
            raw_action, action = policy.step(
                rgb_frame,
                INSTRUCTION,
            )

            # Read current pose after inference, immediately before target creation.
            current_tcp = list(rtde_r.getActualTCPPose())

            # Adapter should clamp translation and hold orientation fixed.
            target_tcp = openvla_to_ur5_target(
                current_tcp,
                action,
            )

            # is_safe = rtde_c.isPoseWithinSafetyLimits(target_tcp) and rtde_c.getInverseKinematicsHasSolution(target_tcp)
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

            if not pose_is_safe:
                raise RuntimeError(
                    "Target pose was rejected by configured UR safety limits."
                )
            
            if not ik_exists:
                raise RuntimeError(
                    "Target pose does not have a valid inverse-kinematics solution."
                )

            if REQUIRE_CONFIRMATION:
                confirm_move(step_number)

            print("\nExecuting one slow moveL()...")
            moved = rtde_c.moveL(
                target_tcp,
                SPEED_M_PER_S,
                ACCEL_M_PER_S2,
            )

            print("moveL returned:", moved)
            print("TCP pose after move:", rtde_r.getActualTCPPose())

            if step_number < MAX_STEPS:
                print(
                    f"\nWaiting {INTER_STEP_PAUSE_SECONDS:.1f} seconds "
                    "before the next image and inference step..."
                )
                time.sleep(INTER_STEP_PAUSE_SECONDS)

        print("\nCompleted all configured supervised motion steps.")

    except KeyboardInterrupt as exc:
        print(f"\nStopped: {exc}")

        if rtde_c is not None:
            print("Sending stopL()...")
            rtde_c.stopL()

    except Exception as exc:
        print(f"\nMovement test failed: {exc}")

        if rtde_c is not None:
            print("Sending stopL()...")
            rtde_c.stopL()

        sys.exit(1)

    finally:
        if capture is not None:
            capture.release()

        if rtde_c is not None:
            rtde_c.stopScript()

        print("\nConnections closed.")


if __name__ == "__main__":
    main()