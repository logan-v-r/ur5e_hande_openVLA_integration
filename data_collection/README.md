# Data Collection

This directory contains the script used to record UR5e demonstrations for the custom OpenVLA fine-tuning dataset.

## Contents

| File                      | Purpose                                                                                                                  |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `collect_data_gripper.py` | Records RealSense images, UR5e telemetry, derived actions, gripper information, task instructions, and episode metadata. |

## How It Works

`collect_data_gripper.py` runs alongside [`mirror_relative_keyboard.py`](../robot_control/mirror_relative_keyboard.py).

The mirror script controls the UR5e by:

* Reading relative movement from a manually operated UR7e
* Transforming and mirroring that movement onto the UR5e
* Accepting keyboard commands for the Hand-E gripper
* Optionally writing the current gripper command to a shared status file

The collection script is read-only. It does not command the robot or gripper. It records the UR5e demonstration in a separate terminal while the mirror script performs the control.

## Data Collection Procedure

A typical demonstration is collected as follows:

1. Move both robots to their saved starting positions.
2. Arrange the workspace for the selected task instruction.
3. Confirm that both operators, the camera, robots, and gripper are ready.
4. Run `mirror_relative_keyboard.py` in the first terminal.
5. Run `collect_data_gripper.py` in a separate terminal.
6. Perform the demonstration by moving the UR7e and entering gripper commands.
7. Stop the recorder cleanly when the task is complete.
8. Review the episode and remove it if the demonstration was unsuccessful or inaccurate.

## Recorded Output

Each raw episode contains:

```text
episode_.../
├── episode_metadata.json
├── steps.jsonl
├── images/
└── COMPLETE.json
```

The recorded data includes:

* RGB camera images
* UR5e TCP and joint states
* Relative translation and rotation actions calculated between observations
* Gripper state or command, when a configured source is available
* Natural-language task instruction
* Timing, camera, robot, and quality metadata

`COMPLETE.json` is written only after a clean shutdown. The raw episodes are later reviewed and processed using [`clean_raw_episodes.py`](../data_processing/clean_raw_episodes.py).

Raw demonstrations are not committed to this repository because of their size.

## Workflow

```text
`mirror_relative_keyboard.py`
              +
 `collect_data_gripper.py`
              ↓
      Raw demonstration episodes
              ↓
       Manual task review
              ↓
      `clean_raw_episodes.py`
              ↓
       TFDS/RLDS conversion
```

## Safety

This workflow controls physical industrial robots and requires active supervision. Keep the emergency stop accessible, clear the workspace before movement, and stop immediately if the UR5e does not follow the expected motion.
