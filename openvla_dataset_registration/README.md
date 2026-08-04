# OpenVLA Dataset Registration

This directory contains the OpenVLA data-loader changes required to use the custom `ur5e_openvla` RLDS dataset for fine-tuning.

Converting the dataset to RLDS format is not enough by itself. OpenVLA also requires the dataset to be registered in its OXE configuration and transform registries.

## Files

### `configs.py`

Project-modified version of:

```text
prismatic/vla/datasets/rlds/oxe/configs.py
```

It adds the `ur5e_openvla` entry to `OXE_DATASET_CONFIGS` and tells OpenVLA how to interpret the dataset fields.

The configuration specifies:

* `image` as the primary RGB observation;
* `EEF_state` as the six-value end-effector state;
* `gripper_state` as the physical gripper observation;
* `ActionEncoding.EEF_POS` for relative translation and rotation actions plus one absolute gripper target.

The UR5e orientation values are stored as rotation vectors `[rx, ry, rz]`, not conventional Euler angles. `StateEncoding.POS_EULER` is used only because OpenVLA expects a six-value end-effector state layout.

### `transforms.py`

Project-modified version of:

```text
prismatic/vla/datasets/rlds/oxe/transforms.py
```

It adds:

```python
ur5e_openvla_dataset_transform
```

and registers it under:

```text
ur5e_openvla
```

The raw RLDS action is:

```text
[dx, dy, dz, drx, dry, drz, gripper_closed_target]
```

with:

```text
0 = open
1 = closed
```

The transform converts the final action dimension to OpenVLA’s expected convention:

```text
[dx, dy, dz, drx, dry, drz, gripper_open_target]
```

with:

```text
1 = open
0 = closed
```

The first six action dimensions are unchanged. The observation gripper state is also preserved as `0=open, 1=closed`; only the action target is inverted.

## Adding the Files to OpenVLA

The corresponding OpenVLA files are located at:

```text
prismatic/vla/datasets/rlds/oxe/configs.py
prismatic/vla/datasets/rlds/oxe/transforms.py
```

The recommended approach is to merge only the `ur5e_openvla` additions into the installed OpenVLA files rather than overwrite the complete files. This avoids removing unrelated upstream updates or dataset registrations.

The dataset name must remain consistent across:

* the RLDS builder;
* the generated TensorFlow dataset;
* `OXE_DATASET_CONFIGS`;
* `OXE_STANDARDIZATION_TRANSFORMS`;
* the fine-tuning command.

Use:

```bash
--dataset_name ur5e_openvla
```

## Workflow

```text
Processed UR5e episodes
        ↓
RLDS dataset builder
        ↓
ur5e_openvla RLDS dataset
        ↓
configs.py and transforms.py registration
        ↓
OpenVLA fine-tuning
```

These files are reference copies of the OpenVLA configuration used by this project. Compare them with the installed OpenVLA version before applying changes.
