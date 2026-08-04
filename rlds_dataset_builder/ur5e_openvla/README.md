# UR5e OpenVLA RLDS Dataset Builder

This directory contains the customized TensorFlow Datasets builder used to convert cleaned UR5e demonstration episodes into RLDS format for OpenVLA fine-tuning.

## Setup

This builder is designed to be used with the upstream [`kpertsch/rlds_dataset_builder`](https://github.com/kpertsch/rlds_dataset_builder) repository.

Clone the upstream repository and follow its README to create and activate the RLDS environment:

```bash
git clone https://github.com/kpertsch/rlds_dataset_builder.git
cd rlds_dataset_builder

conda env create -f environment_ubuntu.yml
conda activate rlds_env
```

Before modifying the example, run the upstream example conversion to confirm the environment is working correctly.

## Add the UR5e Builder

Follow step 1 of the upstream README:

1. Rename the `example_dataset/` directory to `ur5e_openvla/`.
2. Rename:

```text
example_dataset_dataset_builder.py
```

to:

```text
ur5e_openvla_dataset_builder.py
```

3. Replace the contents of the renamed example builder with the `ur5e_openvla_dataset_builder.py` file provided in this directory.

The resulting structure should look like:

```text
rlds_dataset_builder/
├── ur5e_openvla/
│   ├── ur5e_openvla_dataset_builder.py
│   ├── README.md
│   └── ...
├── visualize_dataset.py
├── test_dataset_transform.py
└── ...
```

## What This Builder Already Implements

The provided UR5e builder already performs most of the dataset-specific work described in steps 2–4 of the upstream README:

* defines the RLDS dataset features;
* defines the observation, action, instruction, and metadata fields;
* configures the dataset split;
* reads the cleaned UR5e episode files;
* packages the episode data into RLDS steps;
* implements `_generate_examples()`.

You should not repeat those steps unless the format of the cleaned UR5e data changes.

## Configure the Source Data Path

Before building the dataset, review the builder and confirm that any configured source-data path points to the cleaned UR5e episode directory on your system.

## Build the Dataset

From the renamed dataset directory, run:

```bash
cd rlds_dataset_builder/ur5e_openvla
tfds build --overwrite
```

Unless another TensorFlow Datasets directory is configured, the generated dataset will be written to:

```text
~/tensorflow_datasets/ur5e_openvla/
```

## Verify the Dataset

From the root of the upstream repository, run:

```bash
python3 visualize_dataset.py ur5e_openvla
```

Review the images, language instruction, action values, state values, and episode structure before using the dataset for fine-tuning.

## Scope

This directory provides the customized UR5e dataset-builder implementation. It does not replace the upstream instructions for:

* creating the RLDS environment;
* verifying the example conversion;
* running `tfds build`;
* visualizing and validating the converted dataset;
* configuring optional parallel processing;
* creating and testing a target-spec transform;
* registering the dataset with OpenVLA;
* uploading or publishing the dataset.

Refer to the upstream RLDS Dataset Builder README for those supporting steps.
