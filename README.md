# OpenVLA Integration and Fine-Tuning for UR5e Robotic Manipulation

This repository documents the integration, evaluation, and task-specific fine-tuning of [OpenVLA](https://github.com/openvla/openvla) for a Universal Robots UR5e equipped with a Robotiq Hand-E gripper.

The project is being completed by undergraduate research interns at **Longlab, Atlantic Technological University Galway**. Its primary purpose is to build practical experience with vision-language-action models while investigating how OpenVLA performs when transferred to a robotic setup that differs from the environments represented in its pretraining data.

> **Project status:** Active development. Fine-tuning on the first custom dataset has been completed, and evaluation of the fine-tuned model is currently underway.

---

## Project Overview

Vision-language-action models connect three types of information:

* **Vision:** What the robot observes through a camera
* **Language:** The task instruction provided by the user
* **Action:** The movement and gripper commands produced by the model

OpenVLA receives a natural-language instruction and a single RGB image. It produces a seven-dimensional action:

```text
[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
```

The first three values represent translation, the next three represent rotation, and the final value controls the gripper.

This project adapts that output to the Longlab UR5e setup using the `ur_rtde` Python API. The complete project workflow includes:

1. Evaluating OpenVLA out of the box
2. Collecting task-specific robot demonstrations
3. Reviewing and cleaning the recorded episodes
4. Converting demonstrations into an RLDS-compatible dataset
5. Fine-tuning OpenVLA with LoRA/QLoRA
6. Evaluating the fine-tuned model on the physical UR5e setup

---

## Project Goals

### Learning goal

Develop practical experience with:

* Vision-language-action models
* Physical robot control through UR-RTDE
* Camera-based model inference
* Robot demonstration collection
* RLDS and TensorFlow Datasets
* LoRA-based model fine-tuning
* Physical robot evaluation and troubleshooting

### Engineering goal

Integrate OpenVLA into the existing Longlab robotic workspace and improve its performance on tasks requiring:

* Language grounding
* Object recognition
* Visual reasoning with distractor objects
* Translation and orientation control
* Gripper operation
* Multi-stage pick-and-place behavior

---

## Current Task Set

The first fine-tuning dataset contains demonstrations for the following task groups.

### Move to object

Move the gripper so that it hovers above the object named in the instruction.

The stapler, screwdriver, and pliers may all be visible at the same time, requiring the model to identify the requested object while ignoring distractors.

### Move screwdriver to object

Pick up the screwdriver and place it next to the object named in the instruction. Multiple possible destination objects may be visible in the workspace.

### Place the red block on the yellow platform

Identify the red block, pick it up, and place it on the yellow platform.

### Place the blue block on the red dustpan

Identify the blue block, pick it up, and place it on the red dustpan.

These tasks range from moving toward a language-specified object to completing multi-stage pick-and-place actions.

---

## Hardware Setup

| Component                          | Role                                                                 |
| ---------------------------------- | -------------------------------------------------------------------- |
| Universal Robots UR5e              | Main robot used during demonstration execution and OpenVLA inference |
| Universal Robots UR7e              | Leader device used during demonstration collection                   |
| Robotiq Hand-E                     | Gripper attached to the UR5e                                         |
| Intel RealSense camera             | Provides RGB observations to OpenVLA                                 |
| NVIDIA RTX 4000 Ada Generation GPU | Used for local inference and fine-tuning                             |
| Ubuntu workstation                 | Runs robot-control, dataset, inference, and training software        |
| Ethernet switch/port expander      | Connects the workstation and robots                                  |

The camera is mounted on a tripod and positioned slightly above the UR5e workspace, looking downward toward the robot and task area.

The UR7e is used only during demonstration collection. The UR5e is the robot controlled during both data collection and OpenVLA evaluation.

Exact hardware models and software versions will be added after the lab workstation configuration is verified.

---

## System Architecture

### OpenVLA inference pipeline

```text
Natural-language instruction
             +
       RGB camera image
             │
             ▼
          OpenVLA
             │
             ▼
7D robot action prediction
[translation, rotation, gripper]
             │
             ▼
     UR5e action adapter
             │
       ┌─────┴─────┐
       ▼           ▼
 UR-RTDE arm   Hand-E gripper
   command        command
       │           │
       └─────┬─────┘
             ▼
       UR5e execution
```

OpenVLA receives one image for each predicted action. The action adapter converts the model output into commands suitable for the UR5e and Hand-E gripper.

### Demonstration and fine-tuning pipeline

```text
UR7e operated by hand in teach mode
                 │
                 ▼
 Relative movement calculated from UR7e
                 │
                 ▼
 Coordinate transformation and mirroring
                 │
                 ▼
       Motion executed by UR5e
                 │
       Keyboard gripper commands
                 │
                 ▼
 Images, robot states, actions, and metadata recorded
                 │
                 ▼
        Raw demonstration episodes
                 │
                 ▼
      Episode review and data cleaning
                 │
                 ▼
       Cleaned demonstration episodes
                 │
                 ▼
       Custom TFDS/RLDS dataset builder
                 │
                 ▼
          `ur5e_openvla` dataset
                 │
                 ▼
       OpenVLA LoRA/QLoRA fine-tuning
                 │
                 ▼
       Fine-tuned model evaluation
```

---

## Repository Structure

```text
ur5e_hande_openVLA_integration/
├── robot_control/          # Robot communication, motion mirroring, and gripper control
├── data_collection/        # Demonstration recording tools
├── data_processing/        # Demonstration cleaning and validation
├── rlds_dataset_builder/   # Custom TFDS/RLDS dataset conversion
├── training/               # OpenVLA fine-tuning scripts and configuration
├── inference/              # Model inference and physical robot execution
├── docs/                   # Extended architecture, evaluation, and setup documentation
├── .gitignore
├── LICENSE
└── README.md
```

### Directory responsibilities

| Directory                                        | Purpose                                                                                                                        |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| [`robot_control/`](robot_control/)               | Connects to the robots, mirrors relative UR7e movement onto the UR5e, and controls the Hand-E gripper.                         |
| [`data_collection/`](data_collection/)           | Records images, robot states, actions, task instructions, and metadata during demonstrations.                                  |
| [`data_processing/`](data_processing/)           | Reviews, cleans, validates, and prepares raw demonstration episodes for dataset conversion.                                    |
| [`rlds_dataset_builder/`](rlds_dataset_builder/) | Converts cleaned demonstrations into the custom `ur5e_openvla` TFDS/RLDS dataset.                                              |
| [`training/`](training/)                         | Contains the scripts and configuration used to fine-tune OpenVLA on the custom dataset.                                        |
| [`inference/`](inference/)                       | Loads OpenVLA, adapts its actions for the UR5e, executes robot commands, and records evaluation data.                          |
| [`docs/`](docs/)                                 | Provides extended documentation for the system architecture, dataset format, evaluation protocol, safety, and troubleshooting. |

Each directory contains its own README with detailed file descriptions, configuration requirements, inputs, outputs, and example commands.

---

## Project Workflow

### 1. Out-of-the-box evaluation

OpenVLA was first connected to the UR5e through a custom inference and action-adaptation pipeline.

The initial testing progressed through three control configurations:

* Translation only
* Translation and rotation
* Translation, rotation, and gripper control

These tests were used to confirm that the camera, model, action adapter, and robot-control pipeline worked together while identifying the main limitations of the out-of-the-box model.

### 2. Demonstration collection

A UR7e was operated manually in teach mode and used as a physical leader device.

The relative change in the UR7e pose was calculated, transformed to account for the physical arrangement of the robots, and mirrored onto the UR5e. The operator watched the UR5e while moving the UR7e so that demonstrations were based on the behavior of the robot used during inference.

Hand-E gripper commands were entered separately through the keyboard.

During each demonstration, the system recorded information such as:

* RGB camera images
* UR5e TCP poses
* Relative translation and rotation actions
* Gripper state or command
* Natural-language task instruction
* Timing and episode metadata

Unsuccessful or noticeably imprecise demonstrations were removed before dataset conversion.

### 3. Data cleaning and validation

The retained raw episodes were processed using `clean_raw_episodes.py`.

The cleaning process removes invalid or low-information steps, preserves useful context around meaningful movement, recalculates actions between retained poses, and produces cleaned copies without modifying the original recordings.

The cleaned episodes were then converted into the custom `ur5e_openvla` TensorFlow dataset.

Dataset verification was performed using `visualize_dataset.py` from the [`rlds_dataset_builder`](https://github.com/kpertsch/rlds_dataset_builder) repository.

Verification included:

* Viewing the beginning, middle, and end of randomly selected episodes
* Confirming that image sequences represented the intended robot motion
* Checking that language instructions matched the selected episodes
* Reviewing the distribution of translation, rotation, and gripper actions
* Identifying unexpected values before fine-tuning

The raw demonstrations, generated TensorFlow dataset, and visualization outputs are not stored in this repository because of their size.

### 4. Fine-tuning

OpenVLA 7B was fine-tuned on the custom `ur5e_openvla` dataset using a LoRA/QLoRA-based workflow.

The training process followed the general approach described in the official OpenVLA repository, with modifications for the local dataset and RTX 4000 Ada Generation workstation.

The first full training run used approximately the following configuration:

| Setting                     | Value                |
| --------------------------- | -------------------- |
| Base model                  | `openvla/openvla-7b` |
| Dataset                     | `ur5e_openvla`       |
| LoRA rank                   | `8`                  |
| Per-device batch size       | `1`                  |
| Gradient accumulation steps | `16`                 |
| Effective batch size        | `16`                 |
| Learning rate               | `5e-4`               |
| Image augmentation          | Disabled             |
| Quantization                | Enabled              |
| Maximum training steps      | `1000`               |
| Checkpoint interval         | Every `250` steps    |

Model weights, adapter outputs, checkpoints, and generated datasets are intentionally excluded from version control.

### 5. Fine-tuned evaluation

Fine-tuning on the first dataset has been completed.

The resulting model is currently being evaluated on the same UR5e setup. The goal is to compare its performance with the out-of-the-box baseline using controlled object positions, task instructions, and success criteria.

---

## Out-of-the-Box Results

The current evaluation spreadsheets document **50 out-of-the-box trials**:

| Test mode                          | Trials | Steps per trial | Example instruction          |
| ---------------------------------- | -----: | --------------: | ---------------------------- |
| Translation only                   |     20 |              20 | “Move towards the red block” |
| Translation and rotation           |     15 |              15 | “Move toward the red block”  |
| Translation, rotation, and gripper |     15 |              15 | “Pick up the red block”      |

One camera image was saved for each inference step so the robot trajectory could be reviewed after testing.

The trials have not yet been formally labeled as successes or failures, so success rates are not currently reported.

Preliminary observations include:

* Translation-only control sometimes moved the robot in approximately the correct direction.
* Behavior became substantially less predictable when rotation was enabled.
* Some trials ended with inverse-kinematics or path-sanity errors.
* Reliable grasping and complete pick-and-place behavior were not achieved out of the box.
* In some trials, the robot initially approached the target and later moved away from it.
* The baseline results supported the need for workspace- and task-specific fine-tuning.

Detailed evaluation procedures and future trial-level results will be documented separately in `docs/evaluation.md`.

---

## Installation Overview

This repository assumes a general understanding of:

* Python and Linux
* Universal Robots
* UR-RTDE
* OpenVLA
* TensorFlow Datasets and RLDS
* GPU-based model inference and training

It does not replace the official setup instructions for its major dependencies.

### OpenVLA

Follow the installation and environment setup instructions in the official [OpenVLA repository](https://github.com/openvla/openvla).

OpenVLA should be installed and tested independently before it is connected to a physical robot.

### RLDS dataset builder

The custom dataset conversion workflow follows the structure described in the [`rlds_dataset_builder`](https://github.com/kpertsch/rlds_dataset_builder) repository.

### Robot and camera dependencies

The project also requires:

* `ur_rtde`
* Intel RealSense software and Python bindings
* Robotiq Hand-E communication support
* TensorFlow
* TensorFlow Datasets
* PyTorch and the OpenVLA dependencies

Exact tested versions will be added after the lab workstation configuration is verified.

| Dependency          | Version |
| ------------------- | ------- |
| Ubuntu              | `TBD`   |
| Python              | `TBD`   |
| CUDA                | `TBD`   |
| PyTorch             | `TBD`   |
| Transformers        | `TBD`   |
| TensorFlow          | `TBD`   |
| TensorFlow Datasets | `TBD`   |
| `ur_rtde`           | `TBD`   |
| RealSense SDK       | `TBD`   |

Detailed installation and execution instructions will be provided in the relevant subdirectory READMEs.

---

## Safety

> **Warning:** This project controls physical industrial robot arms. Incorrect commands can cause collisions, equipment damage, or personal injury.

The software is a research prototype and is not intended for unattended or production use.

Recommended precautions include:

* Keep the emergency stop accessible.
* Maintain a clear workspace.
* Keep personnel outside the robot’s reachable area during autonomous motion.
* Begin with low speeds and conservative action limits.
* Test translation, rotation, and gripper behavior separately.
* Prefer single-step execution before enabling repeated inference.
* Stop testing immediately when movement becomes unstable or unexpected.
* Review the cause of protective stops before continuing.
* Keep someone actively monitoring the robot during execution.

OpenVLA may generate unsafe or nonsensical actions, especially when the physical environment differs from its training data.

---

## Current Limitations

* The project is still under active development.
* Fine-tuned evaluation is not yet complete.
* The custom dataset is relatively small and specific to one lab setup.
* The model is sensitive to camera position, object placement, and workspace appearance.
* The OOTB trials have not yet been assigned formal success labels.
* Some configuration values remain embedded in local scripts.
* Full reproduction requires compatible robot arms, a gripper, a camera, and a sufficiently capable GPU.
* Raw demonstrations, generated datasets, and model weights are not included in the repository.
* The system has not been validated for unattended operation.

---

## Future Work

Planned next steps include:

* Complete the post-fine-tuning evaluation
* Define consistent success criteria for every task
* Review and formally label the OOTB trials
* Compare OOTB and fine-tuned success rates
* Investigate failures during grasping and placement
* Improve rotation handling
* Evaluate different inference frequencies and action scales
* Collect additional demonstrations for weak tasks
* Move hardware and model settings into configuration files
* Add tests for action conversion and gripper mapping
* Add architecture diagrams and demonstration media
* Document the final software environment

---

## Contributions

This project was completed by undergraduate robotics research interns **Alex Ospina** and **Logan Rahner** at Longlab, Atlantic Technological University Galway.

### Alex Ospina

Primary contributions include:

* Setting up OpenVLA and its inference environment using the official OpenVLA instructions
* Connecting live camera input to OpenVLA
* Connecting OpenVLA inference to the UR5e through `ur_rtde`
* Adapting an OpenVLA inference wrapper from [SimplerEnv-OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA)
* Developing and testing the UR5e action-adaptation pipeline
* Evaluating out-of-the-box OpenVLA behavior
* Developing the UR7e-to-UR5e relative motion-mirroring workflow
* Integrating keyboard control of the Hand-E gripper during data collection
* Collecting and cleaning demonstration episodes
* Converting demonstrations using the RLDS dataset-builder workflow
* Modifying the fine-tuning process for the local RTX 4000 workstation
* Fine-tuning OpenVLA on the custom `ur5e_openvla` dataset

### Logan Rahner

Primary contributions include:

* Assisting with out-of-the-box testing
* Collecting demonstrations
* Reviewing and cleaning recorded episodes
* Logging robot and model behavior
* Evaluating experimental performance
* Setting up the OpenVLA environment after fine-tuning
* Helping to connect and calibrate UR5e through `ur_rtde`
* Proofreading code and performing sanity-checks
* Configuring UR5e and UR7e start positions and quick-reset programs for episode collection
* Designed training tasks
* Leading the current post-fine-tuning evaluation work

Both researchers worked together during physical demonstration collection, robot testing, troubleshooting, and evaluation.

---

## Acknowledgements

This work was completed at **Longlab, Atlantic Technological University Galway** as part of an undergraduate robotics research internship.

The project builds on:

* [OpenVLA](https://github.com/openvla/openvla)
* [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
* [SimplerEnv-OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA)
* [RLDS Dataset Builder](https://github.com/kpertsch/rlds_dataset_builder)
* [UR-RTDE](https://sdurobotics.gitlab.io/ur_rtde/)
* [TensorFlow Datasets](https://www.tensorflow.org/datasets)

---

## License

This project is available under the terms described in the repository’s [LICENSE](LICENSE) file.
