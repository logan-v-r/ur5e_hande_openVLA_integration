"""
Run supervised multi-step OpenVLA inference on the physical UR5e.

For each inference step, this script:

1. Captures and saves a fresh RGB camera observation.

2. Sends the image and language instruction to the fine-tuned OpenVLA model.

3. Receives an eight-dimensional action:

   ```
   [dx, dy, dz, drx, dry, drz, gripper_delta, terminate]
   ```

4. Stops the episode without executing an action when the model predicts
   the terminal action.

5. Uses `ur5_action_adapter.py` to:
   - convert the relative arm action into a UR5e TCP target;
   - convert the gripper-state delta into "open", "close", or None.

6. Checks the TCP target using the UR controller's safety and
   inverse-kinematics functions.

7. Executes an asynchronous linear movement while displaying a live
   camera preview.

8. Executes a gripper command after the arm movement when one is predicted.

9. Captures a new observation and repeats.

Responsibilities are separated across three files:

```
openvla_inference.py
    Loads the fine-tuned model, de-normalizes its output, and formats
    the custom eight-dimensional action.

ur5_action_adapter.py
    Converts the processed base-frame action into a limited UR5e TCP
    target and a high-level gripper command. It does not command hardware.

openvla_move_with_liveview.py
    Connects to the UR5e, camera, and Hand-E gripper; performs safety
    checks; and executes the resulting commands.
```

## Saved data

The script saves the exact camera image used for each inference step and
appends one structured record per step to a JSON Lines file.

Output is written under `OUTPUT_DIR`:

```
OUTPUT_DIR/
├── inference_actions.jsonl
├── inference_step_001_<timestamp>.jpg
├── inference_step_002_<timestamp>.jpg
└── ...
```

Each JPEG contains the RGB observation supplied to OpenVLA for that step.

Each line of `inference_actions.jsonl` is an independent JSON object. A
record may include:

```
step:
    One-based inference-step number.

timestamp:
    Date and time when the step was processed.

image_path:
    Path to the saved camera observation.

instruction:
    Natural-language instruction supplied to OpenVLA.

current_tcp:
    UR5e TCP pose before execution.

raw_action:
    De-normalized eight-dimensional model output.

processed_action:
    Action fields passed to the UR5e action adapter.

target_tcp:
    Proposed UR5e TCP target, when an arm action is evaluated.

gripper_command:
    "open", "close", or None.

terminate_predicted:
    Whether the terminal-action threshold was reached.

pose_is_safe:
    Result of the UR controller's pose-safety check.

ik_exists:
    Whether an inverse-kinematics solution exists for the target.

resulting_tcp:
    UR5e TCP pose after successful execution.

execution_status:
    Outcome such as "executed", "stopped_on_terminal_prediction",
    "rejected_by_safety_limits", or "rejected_no_ik_solution".
```

JSON Lines is used so that each step is written immediately and can be
read independently, even if a later step is interrupted. Existing output
is appended to rather than automatically replaced. Use a new or cleared
`OUTPUT_DIR` for each trial when separate trial records are required.

## Program execution

Run the script from the repository root so that the `inference` and
`robot_control` package imports resolve correctly:

```
python -m inference.openvla_move_with_liveview
```

Before running, update the configuration constants near the top of this
file, especially:

```
MODEL_PATH
    Local merged fine-tuned model directory containing
    `dataset_statistics.json`.

UNNORM_KEY
    Dataset normalization key. Leave as None for automatic selection
    when the model contains `ur5e_openvla` or only one available key.

INSTRUCTION
    Natural-language task instruction.

ROBOT_IP
    Network address of the UR5e.

CAMERA_INDEX
    Linux/OpenCV camera index for the RealSense RGB stream.

OUTPUT_DIR
    Directory where inference images and the JSONL log will be saved.
```

Example configuration:

```
MODEL_PATH = os.path.expanduser(
    "~/workspaces/openvla/runs/ur5e_openvla_finetuned"
)

UNNORM_KEY = "ur5e_openvla"
INSTRUCTION = "place the red block on the yellow platform"
ROBOT_IP = "192.168.1.102"
CAMERA_INDEX = 4

OUTPUT_DIR = Path(
    os.path.expanduser(
        "~/workspaces/openvla/logs/"
        "red_block_yellow_platform/trial_01"
    )
)
```

Example execution:

```
cd ~/path/to/ur5e_hande_openVLA_integration

python -m inference.openvla_move_with_liveview
```

For a new trial, change `OUTPUT_DIR` to a new trial directory before
running again. This keeps the saved images and JSONL records from
different evaluations separate.

## Safety

This is supervised research software. It is not intended for unattended
or production robot operation.

Before execution:

* confirm that the correct model and task instruction are configured;
* verify the robot IP address and camera index;
* place the UR5e in the expected starting configuration;
* confirm that the workspace matches the intended task;
* keep the emergency stop accessible;
* keep personnel outside the robot's reachable workspace; and
* begin with conservative action limits and a small value of `MAX_STEPS`.
  """


from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation

from inference import ur5_action_adapter as action_adapter
from inference.openvla_inference import OpenVLAInference
from inference.ur5_action_adapter import (
    openvla_to_gripper_command,
    openvla_to_ur5_target,
)
from robot_control import robotiq_gripper

import traceback


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Local directory containing the merged fine-tuned model and its
# dataset_statistics.json file.
MODEL_PATH = os.path.expanduser(
    "/home/atu-2/workspaces/openvla/runs/openvla-7b+ur5e_openvla+b16+lr-0.0005+lora-r8+dropout-0.0+q-4bit--ur5e_qlora_r8_b1_acc16"
)

# Leave as None to allow OpenVLAInference to select `ur5e_openvla` or the
# only available normalization-statistics key.
UNNORM_KEY = "ur5e_openvla" # str | None = None

# Optional multiplier applied by OpenVLAInference to translation and rotation.
# Gripper and terminal values are not scaled.
ACTION_SCALE = 1.0

INSTRUCTION = "Place the red block on the yellow platform"

# Stop the episode when the predicted terminal value reaches this threshold.
TERMINATION_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# UR5e configuration
# ---------------------------------------------------------------------------

# This value should correspond with YOUR robot's IP!
ROBOT_IP = "192.168.1.102"

# Maximum number of image-action cycles allowed if the model does not
# predict termination.
MAX_STEPS = 60

# Pause between completed actions and the next camera observation.
INTER_STEP_PAUSE_SECONDS = 0.05

# Conservative linear movement settings.
SPEED_M_PER_S = 0.1
ACCEL_M_PER_S2 = 0.8

# Deceleration used when stopping an asynchronous linear movement.
STOP_ACCEL_M_PER_S2 = 0.5

# Maximum time allowed for one arm movement.
MOVE_TIMEOUT_SECONDS = 10.0

# Pose tolerances used to determine whether the asynchronous movement
# has reached its target.
POSITION_TOLERANCE_METERS = 0.0005
ORIENTATION_TOLERANCE_RADIANS = 0.005


# ---------------------------------------------------------------------------
# Camera configuration
# ---------------------------------------------------------------------------

# Update this after checking the available Linux video devices.
CAMERA_INDEX = 4

SHOW_LIVE_PREVIEW = True
PREVIEW_WINDOW_NAME = "OpenVLA UR5e Live Feed"

# Number of frames discarded when the camera is first opened so exposure
# and white balance can stabilize.
CAMERA_WARMUP_FRAMES = 60


# ---------------------------------------------------------------------------
# Hand-E gripper configuration
# ---------------------------------------------------------------------------

# Keep this synchronized with action_adapter.USE_GRIPPER.
USE_GRIPPER = True

GRIPPER_IP = ROBOT_IP
GRIPPER_PORT = 63352 # Your port may be different!

GRIPPER_SPEED = 64
GRIPPER_FORCE = 20

GRIPPER_OPEN_POSITION = 0
GRIPPER_CLOSED_POSITION = 255


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(
    os.path.expanduser(
        "~/workspaces/openvla/ur5_rtde/logs/"
        "fine_tuned_openvla/current_trial"
    )
)

ACTION_LOG_FILENAME = "inference_actions.jsonl"


def to_jsonable(value: Any) -> Any:
    """
    Convert NumPy values and nested structures into JSON-safe objects.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    return value


def append_action_log(record: dict[str, Any]) -> None:
    """
    Append one inference-step record to the JSONL action log.
    """
    log_path = OUTPUT_DIR / ACTION_LOG_FILENAME

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(
            json.dumps(
                to_jsonable(record),
                allow_nan=False,
            )
            + "\n"
        )


def save_frame(
    bgr_frame: np.ndarray,
    step_number: int,
) -> Path:
    """
    Save the exact camera frame used for one OpenVLA inference step.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    image_path = OUTPUT_DIR / (
        f"inference_step_{step_number:03d}_{timestamp}.jpg"
    )

    if not cv2.imwrite(str(image_path), bgr_frame):
        raise RuntimeError(
            f"Could not save camera image to {image_path}."
        )

    return image_path


def get_camera_frame(
    capture: cv2.VideoCapture,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read one camera observation.

    Returns:
        bgr_frame:
            OpenCV BGR image used for display and logging.

        rgb_frame:
            RGB uint8 image sent to OpenVLA.
    """
    ok, bgr_frame = capture.read()

    if not ok or bgr_frame is None:
        raise RuntimeError(
            "Could not read a frame from the RealSense camera."
        )

    if bgr_frame.dtype != np.uint8:
        raise RuntimeError(
            f"Camera returned unexpected dtype {bgr_frame.dtype}."
        )

    rgb_frame = cv2.cvtColor(
        bgr_frame,
        cv2.COLOR_BGR2RGB,
    )

    return bgr_frame, rgb_frame


def rotation_error_radians(
    actual_tcp: np.ndarray,
    target_tcp: np.ndarray,
) -> float:
    """
    Calculate orientation error using rotation composition.

    UR TCP orientations are rotation vectors. They should not be
    subtracted component-by-component because different rotation vectors
    can represent equivalent or closely related orientations.
    """
    actual_rotation = Rotation.from_rotvec(actual_tcp[3:6])
    target_rotation = Rotation.from_rotvec(target_tcp[3:6])

    relative_rotation = target_rotation.inv() * actual_rotation

    return float(relative_rotation.magnitude())


def show_preview_frame(
    bgr_frame: np.ndarray,
    status_text: str,
    elapsed_seconds: float | None = None,
) -> int:
    """
    Display one annotated camera-preview frame.

    Returns:
        The pressed key code, or -1 when no key was pressed.
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

    cv2.imshow(
        PREVIEW_WINDOW_NAME,
        preview_frame,
    )

    return cv2.waitKey(1) & 0xFF


def connect_gripper() -> robotiq_gripper.RobotiqGripper:
    """
    Connect to and activate the Robotiq Hand-E gripper.

    Automatic calibration is disabled because the driver's complete
    calibration routine opens, closes, and reopens the gripper.
    """
    print(
        "Connecting to Robotiq gripper at "
        f"{GRIPPER_IP}:{GRIPPER_PORT}..."
    )

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
    Execute one high-level Hand-E gripper command.

    The command is produced from the model's predicted gripper-state
    delta:

        positive delta -> "close"
        negative delta -> "open"
        no-change deadband -> None

    Every non-None delta command is executed. Commands are not suppressed
    based on the previous command because the model predicts state changes,
    not an absolute desired gripper state.
    """
    if command is None:
        print("No gripper-state change predicted.")
        return

    if command == "open":
        position = GRIPPER_OPEN_POSITION
    elif command == "close":
        position = GRIPPER_CLOSED_POSITION
    else:
        raise ValueError(
            f"Unsupported gripper command: {command!r}"
        )

    print(
        f"Executing gripper command: {command} "
        f"(position={position})"
    )

    final_position, object_status = (
        gripper.move_and_wait_for_pos(
            position,
            GRIPPER_SPEED,
            GRIPPER_FORCE,
        )
    )

    print(
        "Gripper result:",
        f"position={final_position},",
        f"status={object_status.name}",
    )


def validate_target_pose(
    rtde_c: rtde_control.RTDEControlInterface,
    target_tcp: list[float],
) -> tuple[bool, bool]:
    """
    Check the proposed target using the UR controller.

    Returns:
        pose_is_safe:
            Whether the target is within the configured UR safety limits.

        ik_exists:
            Whether the controller can find an inverse-kinematics solution.
    """
    pose_is_safe = bool(
        rtde_c.isPoseWithinSafetyLimits(target_tcp)
    )

    ik_exists = bool(
        rtde_c.getInverseKinematicsHasSolution(target_tcp)
    )

    return pose_is_safe, ik_exists


def execute_move_with_live_preview(
    rtde_c: rtde_control.RTDEControlInterface,
    rtde_r: rtde_receive.RTDEReceiveInterface,
    capture: cv2.VideoCapture,
    target_tcp: list[float],
) -> None:
    """
    Execute one asynchronous linear movement.

    While the robot is moving, the camera preview remains active. Press
    Q or Esc while the preview window is focused to stop the movement.

    The movement also stops when:

    - the target pose is reached;
    - the camera fails; or
    - MOVE_TIMEOUT_SECONDS is exceeded.
    """
    target_tcp_array = np.asarray(
        target_tcp,
        dtype=np.float64,
    )

    print(
        "\nStarting asynchronous moveL() "
        "with live camera preview..."
    )

    movement_accepted = rtde_c.moveL(
        target_tcp,
        SPEED_M_PER_S,
        ACCEL_M_PER_S2,
        True,
    )

    if movement_accepted is False:
        raise RuntimeError(
            "The UR controller did not accept the moveL command."
        )

    move_start_time = time.monotonic()

    while True:
        ok, bgr_frame = capture.read()

        if not ok or bgr_frame is None:
            rtde_c.stopL(STOP_ACCEL_M_PER_S2)
            raise RuntimeError(
                "Could not read a camera frame while the robot "
                "was moving."
            )

        elapsed_seconds = (
            time.monotonic() - move_start_time
        )

        if SHOW_LIVE_PREVIEW:
            key = show_preview_frame(
                bgr_frame,
                "ROBOT MOVING - Press Q or Esc to stop",
                elapsed_seconds,
            )

            if key in (ord("q"), 27):
                print(
                    "\nStop requested from the preview window."
                )

                rtde_c.stopL(STOP_ACCEL_M_PER_S2)

                raise KeyboardInterrupt(
                    "Motion stopped from live preview window."
                )

        actual_tcp = np.asarray(
            rtde_r.getActualTCPPose(),
            dtype=np.float64,
        )

        if actual_tcp.size != 6:
            rtde_c.stopL(STOP_ACCEL_M_PER_S2)
            raise RuntimeError(
                "UR receive interface returned an invalid TCP pose."
            )

        position_error = float(
            np.linalg.norm(
                actual_tcp[:3] - target_tcp_array[:3]
            )
        )

        orientation_error = rotation_error_radians(
            actual_tcp,
            target_tcp_array,
        )

        if (
            position_error <= POSITION_TOLERANCE_METERS
            and orientation_error
            <= ORIENTATION_TOLERANCE_RADIANS
        ):
            print("Robot reached the target pose.")
            return

        if elapsed_seconds > MOVE_TIMEOUT_SECONDS:
            rtde_c.stopL(STOP_ACCEL_M_PER_S2)

            raise TimeoutError(
                "Robot did not reach the target before "
                f"{MOVE_TIMEOUT_SECONDS:.1f} seconds."
            )

        time.sleep(0.01)


def print_step_summary(
    *,
    step_number: int,
    image_path: Path,
    current_tcp: list[float],
    raw_action: dict[str, np.ndarray],
    action: dict[str, np.ndarray],
    target_tcp: list[float] | None,
    gripper_command: str | None,
    terminate_predicted: bool,
    pose_is_safe: bool | None = None,
    ik_exists: bool | None = None,
) -> None:
    """
    Print a readable summary of one inference step.
    """
    print("\nSaved camera image:")
    print(image_path)

    print("\nCurrent TCP pose [x, y, z, rx, ry, rz]:")
    print(current_tcp)

    print("\nRaw fine-tuned OpenVLA action:")
    print(raw_action)

    print("\nProcessed OpenVLA action:")
    print(action)

    print("\nTranslation proposal [dx, dy, dz]:")
    print(action["world_vector"])

    print("\nRotation proposal [drx, dry, drz]:")
    print(action["rot_axangle"])

    print("\nGripper delta:")
    print(action["gripper"])

    print("\nMapped gripper command:")
    print(gripper_command)

    print("\nTerminal prediction:")
    print(action["terminate_episode"])

    print("\nTermination threshold reached:")
    print(terminate_predicted)

    if target_tcp is not None:
        print("\nProposed UR5e target TCP pose:")
        print(target_tcp)

    if pose_is_safe is not None:
        print("\nWithin configured UR safety limits:")
        print(pose_is_safe)

    if ik_exists is not None:
        print("\nInverse-kinematics solution exists:")
        print(ik_exists)


def validate_configuration() -> None:
    """
    Validate configuration values before connecting to hardware.
    """
    if MAX_STEPS < 1:
        raise ValueError("MAX_STEPS must be at least 1.")

    if INTER_STEP_PAUSE_SECONDS < 0.0:
        raise ValueError(
            "INTER_STEP_PAUSE_SECONDS cannot be negative."
        )

    if not 0.0 <= TERMINATION_THRESHOLD <= 1.0:
        raise ValueError(
            "TERMINATION_THRESHOLD must be between 0.0 and 1.0."
        )

    if not INSTRUCTION.strip():
        raise ValueError("INSTRUCTION cannot be empty.")

    if not MODEL_PATH.strip():
        raise ValueError("MODEL_PATH cannot be empty.")

    if USE_GRIPPER != action_adapter.USE_GRIPPER:
        raise ValueError(
            "USE_GRIPPER in this script must match "
            "ur5_action_adapter.USE_GRIPPER."
        )


def main() -> None:
    """
    Run one supervised fine-tuned OpenVLA evaluation episode.

    The terminal prediction is checked before arm or gripper execution.
    This matches the custom training data, whose terminal row contains
    zero arm motion, zero gripper delta, and a terminal value of 1.0.
    """
    validate_configuration()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    capture: cv2.VideoCapture | None = None
    rtde_c: rtde_control.RTDEControlInterface | None = None
    rtde_r: rtde_receive.RTDEReceiveInterface | None = None
    gripper: robotiq_gripper.RobotiqGripper | None = None

    try:
        print(
            f"Connecting to UR5e receive interface at "
            f"{ROBOT_IP}..."
        )

        rtde_r = rtde_receive.RTDEReceiveInterface(
            ROBOT_IP
        )

        print(
            f"Connecting to UR5e control interface at "
            f"{ROBOT_IP}..."
        )

        rtde_c = rtde_control.RTDEControlInterface(
            ROBOT_IP
        )

        if USE_GRIPPER:
            gripper = connect_gripper()

        print(
            f"Opening RealSense camera index "
            f"{CAMERA_INDEX}..."
        )

        capture = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_V4L2,
        )

        if not capture.isOpened():
            raise RuntimeError(
                "Could not open RealSense camera index "
                f"{CAMERA_INDEX}."
            )

        if SHOW_LIVE_PREVIEW:
            cv2.namedWindow(
                PREVIEW_WINDOW_NAME,
                cv2.WINDOW_NORMAL,
            )

        for _ in range(CAMERA_WARMUP_FRAMES):
            ok, _ = capture.read()

            if not ok:
                raise RuntimeError(
                    "Could not read a camera frame "
                    "during warm-up."
                )

        print("\nLoading fine-tuned OpenVLA model...")

        policy = OpenVLAInference(
            saved_model_path=MODEL_PATH,
            unnorm_key=UNNORM_KEY,
            action_scale=ACTION_SCALE,
        )

        policy.reset(INSTRUCTION)

        print("\n" + "=" * 72)
        print("FINE-TUNED OPENVLA → UR5e LIVE INFERENCE")
        print("=" * 72)
        print(f"Instruction: {INSTRUCTION}")
        print(f"Model path: {MODEL_PATH}")
        print(f"Normalization key: {policy.unnorm_key}")
        print(f"Camera index: {CAMERA_INDEX}")
        print(f"Maximum steps: {MAX_STEPS}")
        print(
            "Termination threshold: "
            f"{TERMINATION_THRESHOLD}"
        )
        print(
            "Rotation enabled in adapter: "
            f"{action_adapter.USE_ROTATION}"
        )
        print(
            "Maximum translation per axis: "
            f"{action_adapter.MAX_TRANSLATION_METERS} m"
        )
        print(
            "Maximum rotation magnitude: "
            f"{action_adapter.MAX_ROTATION_RADIANS} rad"
        )
        print(f"Gripper enabled: {USE_GRIPPER}")
        print(
            f"Live preview enabled: "
            f"{SHOW_LIVE_PREVIEW}"
        )

        if SHOW_LIVE_PREVIEW:
            print(
                "Press Q or Esc in the preview window "
                "to stop a movement."
            )

        termination_reached = False

        for step_number in range(
            1,
            MAX_STEPS + 1,
        ):
            print("\n" + "-" * 72)
            print(
                f"STEP {step_number} OF {MAX_STEPS}"
            )
            print("-" * 72)

            bgr_frame, rgb_frame = get_camera_frame(
                capture
            )

            if SHOW_LIVE_PREVIEW:
                key = show_preview_frame(
                    bgr_frame,
                    "OPENVLA INFERENCE - Robot stationary",
                )

                if key in (ord("q"), 27):
                    raise KeyboardInterrupt(
                        "Stopped before inference execution."
                    )

            image_path = save_frame(
                bgr_frame,
                step_number,
            )

            raw_action, action = policy.step(
                rgb_frame,
                INSTRUCTION,
            )

            current_tcp = list(
                rtde_r.getActualTCPPose()
            )

            if len(current_tcp) != 6:
                raise RuntimeError(
                    "UR receive interface returned an "
                    "invalid TCP pose."
                )

            terminate_predicted = (
                policy.should_terminate(
                    action,
                    threshold=TERMINATION_THRESHOLD,
                )
            )

            # The terminal row in the custom dataset contains zero arm
            # motion and zero gripper delta. Stop before calculating or
            # executing a physical command.
            if terminate_predicted:
                print_step_summary(
                    step_number=step_number,
                    image_path=image_path,
                    current_tcp=current_tcp,
                    raw_action=raw_action,
                    action=action,
                    target_tcp=None,
                    gripper_command=None,
                    terminate_predicted=True,
                )

                append_action_log(
                    {
                        "step": step_number,
                        "timestamp": datetime.now().isoformat(),
                        "image_path": str(image_path),
                        "instruction": INSTRUCTION,
                        "current_tcp": current_tcp,
                        "raw_action": raw_action,
                        "processed_action": action,
                        "terminate_predicted": True,
                        "execution_status": (
                            "stopped_on_terminal_prediction"
                        ),
                    }
                )

                print(
                    "\nModel predicted the terminal action. "
                    "Ending the episode."
                )

                termination_reached = True
                break

            target_tcp = openvla_to_ur5_target(
                current_tcp,
                action,
            )

            gripper_command = (
                openvla_to_gripper_command(action)
            )

            pose_is_safe, ik_exists = (
                validate_target_pose(
                    rtde_c,
                    target_tcp,
                )
            )

            print_step_summary(
                step_number=step_number,
                image_path=image_path,
                current_tcp=current_tcp,
                raw_action=raw_action,
                action=action,
                target_tcp=target_tcp,
                gripper_command=gripper_command,
                terminate_predicted=False,
                pose_is_safe=pose_is_safe,
                ik_exists=ik_exists,
            )

            log_record = {
                "step": step_number,
                "timestamp": datetime.now().isoformat(),
                "image_path": str(image_path),
                "instruction": INSTRUCTION,
                "current_tcp": current_tcp,
                "raw_action": raw_action,
                "processed_action": action,
                "target_tcp": target_tcp,
                "gripper_command": gripper_command,
                "terminate_predicted": False,
                "pose_is_safe": pose_is_safe,
                "ik_exists": ik_exists,
            }

            if not pose_is_safe:
                log_record["execution_status"] = (
                    "rejected_by_safety_limits"
                )
                append_action_log(log_record)

                raise RuntimeError(
                    "Target pose was rejected by the "
                    "configured UR safety limits."
                )

            if not ik_exists:
                log_record["execution_status"] = (
                    "rejected_no_ik_solution"
                )
                append_action_log(log_record)

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

            if gripper is not None:
                execute_gripper_command(
                    gripper,
                    gripper_command,
                )

            resulting_tcp = list(
                rtde_r.getActualTCPPose()
            )

            log_record["resulting_tcp"] = resulting_tcp
            log_record["execution_status"] = "executed"

            append_action_log(log_record)

            print("\nTCP pose after action:")
            print(resulting_tcp)

            if step_number < MAX_STEPS:
                print(
                    "\nWaiting "
                    f"{INTER_STEP_PAUSE_SECONDS:.1f} "
                    "seconds before the next observation..."
                )

                time.sleep(
                    INTER_STEP_PAUSE_SECONDS
                )

        if termination_reached:
            print(
                "\nEpisode completed after the model "
                "predicted termination."
            )
        else:
            print(
                "\nReached MAX_STEPS without a terminal "
                "prediction."
            )

    except KeyboardInterrupt as exc:
        print(f"\nStopped: {exc}")

        if rtde_c is not None:
            print("Sending stopL()...")
            rtde_c.stopL(STOP_ACCEL_M_PER_S2)

    except Exception as exc:
        print(f"\nLive inference failed: {exc}")
        traceback.print_exc()

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
