# Data Processing

This directory contains the scripts used to clean and standardize recorded UR5e demonstration episodes before converting them into the custom `ur5e_openvla` RLDS dataset.

## Processing Workflow

```text
Raw demonstration episodes
        ↓
clean_raw_episodes.py
        ↓
Cleaned episodes with relative robot actions
and delta-style gripper commands
        ↓
convert_gripper_delta_to_absolute.py
        ↓
Processed episodes with relative robot actions
and absolute gripper states
        ↓
rlds_dataset_builder/ur5e_openvla/
        ↓
ur5e_openvla RLDS dataset
        ↓
openvla_dataset_registration/
        ↓
OpenVLA fine-tuning
```

The translation and rotation action dimensions remain relative throughout this process. Only the gripper channel is converted from a delta command to an absolute state.

## Files

### `clean_raw_episodes.py`

Cleans retained demonstration episodes and prepares them for RLDS conversion.

The script:

* removes invalid or unusable steps;
* removes extended periods with little or no robot movement;
* preserves context around meaningful movement and gripper events;
* recalculates translation and rotation actions between retained poses;
* copies and reindexes the corresponding images;
* writes cleaned episodes to a separate output directory.

At this stage, the first six action dimensions represent relative end-effector movement:

```text
[dx, dy, dz, drx, dry, drz]
```

The gripper channel may still contain delta-style commands that indicate when the gripper should change state.

### `convert_gripper_delta_to_absolute.py`

Converts the cleaned gripper delta commands into an absolute gripper state at every step.

Instead of retaining only isolated open or close commands, the script tracks the current gripper state and carries it forward through the episode.

The converted data uses:

```text
0 = open
1 = closed
```

The resulting seven-dimensional action is:

```text
[
    dx,
    dy,
    dz,
    drx,
    dry,
    drz,
    gripper_closed_target
]
```

The first six values remain relative robot actions. The final value is an absolute gripper target.

The output of this script should be used as the source data for the `ur5e_openvla` RLDS builder.

## OpenVLA Gripper Convention

The processed episodes and raw RLDS dataset store the gripper as an absolute closed-target value:

```text
0 = open
1 = closed
```

OpenVLA expects the model action to use an absolute open-target convention:

```text
1 = open
0 = closed
```

That final conversion does not occur in this directory. It is handled by the custom OpenVLA dataset transform documented in:

```text
openvla_dataset_registration/
```

The registered `ur5e_openvla_dataset_transform` converts:

```python
gripper_open_target = 1.0 - gripper_closed_target
```

Only the gripper action target is inverted. The first six relative action dimensions pass through unchanged, and the observed physical gripper state remains `0=open, 1=closed`.

The complete gripper workflow is:

```text
Collected delta-style gripper commands
        ↓
convert_gripper_delta_to_absolute.py
        ↓
Absolute closed target:
0 = open, 1 = closed
        ↓
RLDS dataset builder
        ↓
ur5e_openvla_dataset_transform
        ↓
OpenVLA absolute open target:
1 = open, 0 = closed
```

## Running the Scripts

Review the supported command-line arguments before processing data:

```bash
python data_processing/clean_raw_episodes.py --help
python data_processing/convert_gripper_delta_to_absolute.py --help
```

Run `clean_raw_episodes.py` first, followed by `convert_gripper_delta_to_absolute.py`.

Use the converter’s output directory as the input source configured in:

```text
rlds_dataset_builder/ur5e_openvla/ur5e_openvla_dataset_builder.py
```

## Validation

Before building the RLDS dataset, confirm that:

* all retained steps reference valid images;
* translation and rotation actions reflect the intended robot movement;
* every step contains an absolute gripper value;
* the gripper remains open before the grasp;
* the gripper remains closed while carrying the object;
* the gripper returns to open at release;
* language instructions and episode metadata remain associated with the correct episodes.

After RLDS conversion and OpenVLA registration, verify that the final model action uses:

```text
1 = open
0 = closed
```

## Related Directories

* `data_collection/` records raw UR5e demonstrations.
* `data_processing/` cleans episodes and converts gripper deltas to absolute states.
* `rlds_dataset_builder/ur5e_openvla/` converts the processed episodes into RLDS format.
* `openvla_dataset_registration/` registers and standardizes the RLDS dataset for OpenVLA.
* The fine-tuning scripts load the dataset using `--dataset_name ur5e_openvla`.

Raw episodes, processed datasets, and generated RLDS files are not stored in this repository because of their size.
