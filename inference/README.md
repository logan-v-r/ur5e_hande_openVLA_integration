# Inference

This directory contains the tools used to run OpenVLA on the physical UR5e. It loads a fine-tuned `ur5e_openvla` model, converts each predicted action into a UR5e TCP target and Hand-E gripper command, executes the motion, and records the observations and actions used during evaluation.

The same tools were used for out-of-the-box baseline testing and are now used for fine-tuned checkpoint evaluation.

## Inference Workflow

```text
Natural-language instruction
             +
      RGB camera image
             │
             ▼
     openvla_inference.py
   (model loading, de-normalization,
    action formatting)
             │
             ▼
Seven-dimensional processed action
[world_vector, rot_axangle, gripper]
             │
             ▼
   ur5_action_adapter.py
  (per-action limits, target pose,
   gripper command mapping)
             │
             ▼
openvla_move_with_liveview.py
 (safety and IK checks, motion
  execution, gripper execution,
  image and JSONL logging)
             │
             ▼
        UR5e execution
             │
             ▼
     New camera observation
```

One image produces one action. The physical result of that action must be observed in a new image before the next inference step.

## Responsibility Separation

The three files are intentionally separated so that model behaviour, action conversion, and hardware control can be changed and debugged independently.

| File | Responsibility | Touches hardware |
|---|---|---|
| `openvla_inference.py` | Loads the model, prepares the image, de-normalizes the prediction, formats the action | No |
| `ur5_action_adapter.py` | Limits the action and converts it into a TCP target and gripper command | No |
| `openvla_move_with_liveview.py` | Connects to the robot, camera, and gripper; validates and executes commands; logs results | Yes |

## Action Conventions

The fine-tuned `ur5e_openvla` model produces a seven-dimensional action:

```text
[dx, dy, dz, drx, dry, drz, gripper_open_target]
```

**Translation.** `[dx, dy, dz]` is a relative Cartesian displacement expressed in the UR5e base frame, in metres.

**Rotation.** `[drx, dry, drz]` is a relative base-frame rotation vector. The direction of the vector defines the rotation axis and its magnitude defines the rotation angle in radians. These values are already a rotation-vector representation and must not be treated as roll, pitch, and yaw or converted from Euler angles.

**Gripper.** The seventh value is an absolute open target using the OpenVLA convention:

```text
1.0 = open
0.0 = closed
```

This is the inverted form of the closed target stored in the processed episodes and raw RLDS dataset. The inversion is performed by the registered dataset transform, not by this directory. See [`openvla_dataset_registration/`](../openvla_dataset_registration/) and [`data_processing/`](../data_processing/).

## Files

### `openvla_inference.py`

Provides the `OpenVLAInference` class, which loads the merged fine-tuned model and produces one processed action per call.

The class:

* loads the processor and model from a local merged fine-tuned directory;
* loads `dataset_statistics.json` from that directory and attaches it to the model;
* resolves the de-normalization key, selecting `ur5e_openvla` or the only available key when one is not supplied;
* builds the OpenVLA prompt from the language instruction;
* validates the RGB observation and the returned action;
* applies an optional `action_scale` multiplier to translation and rotation;
* returns both the raw model output and the processed action.

The returned pair is:

```text
raw_action
    world_vector
    rotation_delta_base_frame
    gripper_open_target

action
    world_vector
    rot_axangle
    gripper
```

`action` is the dictionary consumed by `ur5_action_adapter.py`. The gripper value is never scaled.

The class also provides `visualize_epoch()`, which saves a figure containing a subsampled image strip and a plot of each of the seven action dimensions across one evaluation episode.

CUDA is required. The model directory must contain `dataset_statistics.json` saved by the fine-tuning run, otherwise de-normalization cannot be performed.

### `ur5_action_adapter.py`

Converts a processed action into a UR5e TCP target and a high-level gripper command. This module does not connect to or command hardware.

`openvla_to_ur5_target(current_tcp_pose, openvla_action)` returns a six-element target pose `[x, y, z, rx, ry, rz]`:

* translation is clipped per Cartesian axis to `MAX_TRANSLATION_METERS`;
* the rotation vector is masked by `ROTATION_AXIS_MASK` and limited by total magnitude to `MAX_ROTATION_RADIANS`;
* the target orientation is reconstructed by left composition:

```text
target_rotation = delta_rotation * current_rotation
```

This is the inverse of the dataset-cleaning calculation:

```text
delta_rotation = target_rotation * inverse(current_rotation)
```

The rotation and translation conventions must remain consistent with `clean_raw_episodes.py` and the custom RLDS builder. When `USE_ROTATION` is `False`, the current orientation is preserved and the predicted rotation delta is ignored.

`openvla_to_gripper_command(openvla_action)` maps the absolute open target to a high-level command using a deadband:

```text
target >= GRIPPER_OPEN_THRESHOLD    -> "open"
target <= GRIPPER_CLOSED_THRESHOLD  -> "close"
otherwise                           -> None
```

Returning `None` leaves the gripper in its current physical state. The deadband prevents uncertain middle-range predictions from repeatedly switching the physical gripper.


### `openvla_move_with_liveview.py`

Runs a supervised multi-step evaluation episode on the physical UR5e.

For each step, the script:

1. discards queued camera frames and captures a fresh RGB observation;
2. saves the exact image supplied to the model;
3. requests one action from `OpenVLAInference`;
4. reads the current TCP pose from the UR receive interface;
5. converts the action into a target pose and gripper command using the adapter;
6. checks the target with `isPoseWithinSafetyLimits()` and `getInverseKinematicsHasSolution()`;
7. executes an asynchronous `moveL()` while displaying a live camera preview;
8. executes the gripper command after the arm movement when one is predicted;
9. appends a structured record to the JSONL action log.

Queued frames are flushed before every observation. This addresses the stale-camera-frame issue identified in Phase 3, where the model received an image that no longer reflected the current robot pose.

The model predicts an absolute gripper target on every cycle. A repeated command is suppressed after the same physical command has already executed successfully, so the Hand-E is not driven to the same position on every step. The first prediction is always sent.

Orientation error is measured by rotation composition rather than component-wise subtraction, because different rotation vectors can represent equivalent orientations.

The movement stops when the target is reached, when `MOVE_TIMEOUT_SECONDS` is exceeded, when the camera fails, or when **Q** or **Esc** is pressed in the focused preview window. On any interruption or failure the script sends `stopL()`, releases the camera, calls `stopScript()`, and disconnects the gripper.

## Configuration

All settings are constants near the top of `openvla_move_with_liveview.py` and `ur5_action_adapter.py`. Review them before every trial.

### Model

```text
MODEL_PATH
    Local merged fine-tuned model directory containing
    dataset_statistics.json.

UNNORM_KEY
    Dataset normalization key. Leave as None for automatic selection when
    the model contains `ur5e_openvla` or only one key is available.

ACTION_SCALE
    Optional multiplier applied to translation and rotation. The gripper
    value is not scaled.

INSTRUCTION
    Natural-language task instruction.
```

### Robot and motion

```text
ROBOT_IP                        Network address of the UR5e
MAX_STEPS                       Maximum image-action cycles in one episode
INTER_STEP_PAUSE_SECONDS        Pause before the next observation
SPEED_M_PER_S                   Linear speed
ACCEL_M_PER_S2                  Linear acceleration
STOP_ACCEL_M_PER_S2             Deceleration used by stopL()
MOVE_TIMEOUT_SECONDS            Maximum time allowed for one movement
POSITION_TOLERANCE_METERS       Target-reached position tolerance
ORIENTATION_TOLERANCE_RADIANS   Target-reached orientation tolerance
```

### Camera

```text
CAMERA_INDEX            Linux/OpenCV index for the RealSense RGB stream
SHOW_LIVE_PREVIEW       Enables the annotated preview window
CAMERA_WARMUP_FRAMES    Frames discarded at start-up for exposure and
                        white balance
CAMERA_FLUSH_FRAMES     Queued frames discarded before each observation
```

### Gripper

```text
USE_GRIPPER                Must match ur5_action_adapter.USE_GRIPPER
GRIPPER_IP                 Defaults to ROBOT_IP
GRIPPER_PORT               Hand-E socket port; yours may differ
GRIPPER_SPEED
GRIPPER_FORCE
GRIPPER_OPEN_POSITION      0
GRIPPER_CLOSED_POSITION    255
```

`validate_configuration()` runs before any hardware connection and rejects an empty instruction, an empty model path, invalid step or pause values, and a `USE_GRIPPER` value that disagrees with the adapter.

Automatic gripper calibration is disabled at connection because the driver's full calibration routine opens, closes, and reopens the gripper.

Example configuration:

```python
MODEL_PATH = os.path.expanduser(
    "~/workspaces/openvla/runs/ur5e_openvla_finetuned"
)

UNNORM_KEY = "ur5e_openvla"
ACTION_SCALE = 1.0
INSTRUCTION = "Place the red block on the yellow platform."
ROBOT_IP = "192.168.1.102"
CAMERA_INDEX = 4

OUTPUT_DIR = Path(
    os.path.expanduser(
        "~/workspaces/openvla/logs/"
        "red_block_yellow_platform/trial_01"
    )
)
```

## Running an Evaluation Episode

Run the script as a module from the repository root so that the `inference` and `robot_control` package imports resolve correctly:

```bash
cd ~/path/to/ur5e_hande_openVLA_integration

python -m inference.openvla_move_with_liveview
```

Change `OUTPUT_DIR` to a new trial directory before each run. Existing output is appended to rather than replaced, so reusing a directory mixes records from separate trials.

## Saved Data

Each run writes the exact camera image used for every inference step and one JSON object per step to a JSON Lines log:

```text
OUTPUT_DIR/
├── inference_actions.jsonl
├── inference_step_001_<timestamp>.jpg
├── inference_step_002_<timestamp>.jpg
└── ...
```

JSON Lines is used so that every step is written immediately and can be read independently, even if a later step is interrupted.

A record may contain:

| Field | Description |
|---|---|
| `step` | One-based inference-step number |
| `timestamp` | Time the step was processed |
| `observation_timestamp` | Time the inference frame was captured |
| `image_path` | Path to the saved observation |
| `instruction` | Instruction supplied to OpenVLA |
| `current_tcp` | TCP pose before execution |
| `raw_action` | De-normalized seven-dimensional model output |
| `processed_action` | Action fields passed to the adapter |
| `target_tcp` | Proposed TCP target |
| `predicted_gripper_command` | Absolute target mapped to `"open"` or `"close"` |
| `gripper_command` | Command actually sent, or `None` when suppressed |
| `pose_is_safe` | Result of the UR pose-safety check |
| `ik_exists` | Whether an IK solution exists for the target |
| `resulting_tcp` | TCP pose after successful execution |
| `execution_status` | `"executed"`, `"rejected_by_safety_limits"`, or `"rejected_no_ik_solution"` |

Saved images and logs support after-the-fact review of a trajectory, comparison of predicted and executed motion, and stage-level scoring of checkpoint behaviour.



## Safety

This workflow controls physical industrial robots and requires active supervision. Keep the emergency stop accessible, clear the workspace before movement, and stop immediately if the UR5e does not follow the expected motion.



## Related Directories

* [`robot_control/`](../robot_control/) provides the Hand-E gripper driver used here.
* [`data_processing/`](../data_processing/) defines the action and gripper conventions this directory must match.
* [`openvla_dataset_registration/`](../openvla_dataset_registration/) performs the gripper inversion that produces the absolute open target predicted at inference time.
* [`training/`](../training/) produces the merged checkpoints and `dataset_statistics.json` loaded by `openvla_inference.py`.
* [`docs/`](../docs/) contains extended evaluation, safety, and troubleshooting documentation.

Model weights, checkpoints, saved images, and inference logs are not stored in this repository because of their size.

