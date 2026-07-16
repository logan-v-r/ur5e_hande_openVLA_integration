# Data Processing

This directory contains the script used to clean raw UR5e demonstration episodes before TFDS/RLDS conversion.

## Contents

| File                    | Purpose                                                                                         |
| ----------------------- | ----------------------------------------------------------------------------------------------- |
| `clean_raw_episodes.py` | Applies automated quality checks and creates cleaned copies of retained demonstration episodes. |

## How It Works

`clean_raw_episodes.py` is run after unsuccessful or inaccurate demonstrations have been removed through manual review.

The script:

* Skips incomplete episodes by default
* Removes steps with invalid camera, TCP, safety, image, or timing data
* Removes long stretches of little or no arm movement
* Treats gripper changes as meaningful actions when gripper data is available
* Preserves a small amount of idle context around meaningful actions
* Recalculates actions between retained UR5e poses
* Reindexes retained steps and images
* Adds a terminal action to the final retained observation by default
* Writes a cleaning summary to `cleaning_manifest.jsonl`

The original raw episodes are not modified. Cleaned episodes are written to a separate output directory and marked as `cleaned_unreviewed`.

## Input and Output

The expected input is the raw episode structure produced by [`collect_data_gripper.py`](../data_collection/collect_data_gripper.py):

```text
episode_.../
├── episode_metadata.json
├── steps.jsonl
├── images/
└── COMPLETE.json
```

The cleaned output preserves the same episode-level structure and also adds cleaning information to the metadata. The output root contains a `cleaning_manifest.jsonl` file summarizing which episodes were cleaned, skipped, or encountered errors.

## Workflow

```text
Raw episodes from `collect_data_gripper.py`
                  ↓
       Manual task-success review
                  ↓
        `clean_raw_episodes.py`
                  ↓
         Cleaned episode folders
                  ↓
          TFDS/RLDS conversion
```

The generated datasets and episode images are not committed to this repository because of their size.

## Validation

After cleaning, the episodes should be checked before dataset conversion to confirm that:

* The expected number of episodes was retained
* Images are present and correctly ordered
* Robot movement is represented accurately
* Gripper actions occur at the expected times
* Language instructions remain associated with the correct episodes

The resulting dataset is later inspected using `visualize_dataset.py` from the [`rlds_dataset_builder`](https://github.com/kpertsch/rlds_dataset_builder) project.

## Related Directories

* [`data_collection/`](../data_collection/) — records raw robot demonstrations
* [`rlds_dataset_builder/`](../rlds_dataset_builder/) — converts cleaned episodes into TFDS/RLDS format
* [`training/`](../training/) — fine-tunes OpenVLA on the generated dataset
