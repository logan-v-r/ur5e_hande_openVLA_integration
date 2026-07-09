# ur5e_hande_openVLA_integration
Repository detailing how we integrated a UR5e robotic arm with a Robotiq hand-E gripper and OpenVLA.

# Overview  
OpenVLA is an open source vision language action (VLA) model that receives a plain language prompt and a single RGB image and ouputs a 7D vector to control the robot arm. For more infomration, go to [OpenVLA](https://openvla.github.io/). This repository specifically deals with the integration of OpenVLA with a UR5e robotic arm attached with a Robotiq Hand-E gripper. For more general information, see [OpenVLA github](https://github.com/openvla/openvla). [CAPABILITIES OF OUR MODEL HERE]. 

# Background  
OpenVLA relies on a 7B parameter agent pretrained on data from labs across the world. Still, it will struggle in most setups without fine-tuning as unique environments, such as the one in the lab this project was created in, are extremely out-of-distribution. Additionally, only about 1% of OpenVLA's pretraining comes from UR5 datasets. This section will outline performance metrics with and without fine-tuning and compare them to the performance of OpenVLA in its original [paper](https://arxiv.org/abs/2406.09246).

## Demos and Performance
### No Fine-tuning
The data for our out-of-the-box testing can be found (sheets link).  
  
The conclusion to be drawn is that OpenVLA indeed struggles without fine-tuning in unique environments. In the paper, OpenVLA thrived out-of-the-box because the lab went to lengths ensuring the environments and camera angles it saw were similar to ones present in pre-training data. 

### After Fine-Tuning
We used LoRA fine-tuning to train our model in our unique environment, as suggested by the OpenVLA github. Information on it can be found on that page, in this [section](https://github.com/openvla/openvla#fine-tuning-openvla-via-lora).   

After fine tuning, the results (show results here once we finish).

# Installation
We operated using python files and UR-RTDE on a Linux system with ubuntu installed, with 20GB RAM. For software dependencies, check our (link to a dependencies folder).  
  
Install OpenVLA according to the instructions in [OpenVLA's github](https://github.com/openvla/openvla#fine-tuning-openvla-via-lora). Install UR-RTDE according to the instructions in (link site where we installed UR-RTDE). Set up your camera (ours was a RealSense ###) and connect to your UR5e. We used wired ethernet connections for both. [INCLUDE DETAILS?] 

## Setup
[INCLUDE TEST FILES AND DESCRIBE STEPS TO GET THINGS RUNNING]
# Fine-Tuning
Here are the steps we used in the process of fine-tuning the model using LoRA. For all questions and other info, look to OpenVLA's [LoRA fine-tuning section](https://github.com/openvla/openvla#fine-tuning-openvla-via-lora).  

First, decide which tasks you wish your UR5e to perform. Then record 50-100 episodes worth of data using a teleoperation method, making sure at minimum to collect TCP actual positon, translation deltas, rotation deltas, gripper status, and camera images at 5-10 Hz.  

Second, clean your collected data by removing failed episodes and scrubbing steps that include zero or near-zero motion.  

Third, convert your data to a format readable by OpenVLA by using the linked RLDS converter [here](https://github.com/kpertsch/rlds_dataset_builder).

[FINISH STEPS REGARDING ACTUAL FINE_TUNING, WEIGHT MANIPULATION, LORA SETTINGS, ETC]


# Troubleshooting
[INCLUDE TESTING FILES WE USED AND SUGGESTIONS]

For any other outstanding questions, the best resource is as always [OpenVLA's github](https://github.com/openvla/openvla).

# References
[INCLUDE REFERENCES IN CHOSEN CITATION FORMAT]



