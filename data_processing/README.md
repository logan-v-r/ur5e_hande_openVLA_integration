# Data Processing

This directory contains the script used to clean and validate raw robot demonstration episodes before converting them into the custom RLDS dataset.

## Contents

| File                    | Purpose                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------ |
| `clean_raw_episodes.py` | Processes raw demonstration episodes and creates cleaned copies suitable for dataset conversion. |

## How It Works

`clean_raw_episodes.py` is run after demonstration collection is complete and poor or unsuccessful demonstrations have been manually removed.

The script reviews each retained episode and:

* Rejects incomplete or invalid episodes
* Removes steps with missing or unusable data
* Removes long periods of little or no robot movement
* Preserves context around meaningful movement and gripper actions
* Recalculates actions between the retained robot poses
* Copies only the images associated with retained steps
* Reindexes the cleaned steps and image files
* Generates a manifest describing the cleaning results

The original raw episodes are not modified. Cleaned episodes are written to a separate output directory.

## Processing Workflow

```text
Raw demonstration episodes
            ↓
 Manual review and removal
   of unsuccessful episodes
            ↓
  `clean_raw_episodes.py`
            ↓
 Cleaned demonstration episodes
            ↓
   TFDS/RLDS dataset conversion
            ↓
      OpenVLA fine-tuning
```

## Input and Output

The script expects a directory containing raw demonstration episodes produced by [`collect_data_gripper.py`](../data_collection/collect_data_gripper.py).

Each cleaned episode retains the data required for later RLDS conversion, including:

* RGB camera images
* Robot poses
* Translation and rotation actions
* Gripper information
* Language instructions
* Timing and episode metadata

Raw and cleaned datasets are not committed to this repository because of their size.

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
