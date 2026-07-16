# Data Collection

This directory contains the script used to record robot demonstrations for the custom OpenVLA fine-tuning dataset.

## Contents

| File                      | Purpose                                                                                                                                 |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `collect_data_gripper.py` | Records camera images, UR5e robot states, actions, gripper information, task instructions, and episode metadata during a demonstration. |

## How It Works

`collect_data_gripper.py` runs alongside [`mirror_relative_keyboard.py`](../robot_control/mirror_relative_keyboard.py).

During collection:

* The UR7e is moved manually in teach mode.
* Relative UR7e movement is transformed and mirrored onto the UR5e.
* The UR5e Hand-E gripper is controlled through keyboard input.
* `collect_data_gripper.py` records the demonstration performed by the UR5e.

The UR7e is used only as the leader device. The UR5e is the robot whose observations and actions are saved because it is also the robot used during OpenVLA inference.

## Data Collection Procedure

A typical demonstration is collected as follows:

1. Move both robots to their saved starting positions.
2. Arrange the workspace for the selected task instruction.
3. Confirm that both operators, the camera, the robots, and the gripper are ready.
4. Run `mirror_relative_keyboard.py` in the first terminal.
5. Run `collect_data_gripper.py` in a separate terminal.
6. Perform the demonstration by moving the UR7e and entering the required gripper commands.
7. Stop the recording when the task is complete.
8. Review the episode and remove it if the demonstration was unsuccessful or inaccurate.

## Recorded Data

Each episode contains the information needed for later cleaning and RLDS conversion, including:

* RGB camera images
* UR5e TCP poses
* Relative translation and rotation actions
* Gripper state or command
* Natural-language task instruction
* Timing and episode metadata

Raw demonstrations are not committed to this repository because of their size.

## Project Workflow

```text
UR7e-to-UR5e motion mirroring
              ↓
     Demonstration recording
              ↓
      Raw episode review
              ↓
   Data cleaning and validation
              ↓
       TFDS/RLDS conversion
              ↓
       OpenVLA fine-tuning
```

Related directories:

* [`robot_control/`](../robot_control/) — motion mirroring and gripper control
* [`data_processing/`](../data_processing/) — raw episode cleaning and validation
* [`rlds_dataset_builder/`](../rlds_dataset_builder/) — RLDS dataset conversion
* [`training/`](../training/) — OpenVLA fine-tuning

## Safety

This workflow controls physical industrial robot arms and requires active supervision.

Before collection:

* Keep the emergency stop accessible.
* Confirm that the workspace is clear.
* Verify that both robots begin in the expected positions.
* Use conservative robot speeds.
* Stop immediately if the UR5e does not mirror the intended movement.
