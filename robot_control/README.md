# Robot Control
This sub directory will contain the files used to control our ur5e + ur7e + robotiq gripper setup.

## Contents

| File                    | Purpose                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------ |
| `live_camera_preview_with_crop.py` | Pulls up a preview of the connected camera's POV that shows the cropped 224 x 224 field of view. |
| `mirror_relative_keyboard.py` | Puts the leader and follower robots into freedrive, moves the follower identically to the leader in relative distance. Can set gripper open/close to a keybind. |
| `robotiq_gripper.py` | Includes all infrastructure needed to communicate with and control the Hand-E gripper. |

## File Information

  `live_camera_preview_with_crop.py` pulls up a camera preview that shows the final 224x224 image as well as the unchanged camera view with a green box that indcates what will be included in the final crop. Our lab used a Realsense Camera connected via USB. A similar setup may need to tweak the CAMERA_INDEX variable to get an image (it is usually a number 0-4). A setup that uses streaming or other kinds of connection may need to change the top part of the code that establishes connection with the camera.

  `mirror_relative_keyboard.py` allows teleoperation through leader/follower freedrive control, basically meaning one robot is guided by hand in freederive and the other follows. Because it maps movements relatively, you can use any robotic arms as leader/follower as long as they understand UR RTDE. Important variable for configuration include LEADER_IP, FOLLOWER_IP, FRAME_TRANSFORM, and GRIPPER_PORT. GRIPPER_PORT is the port number of the port where it is plugged into the robot. FRAME_TRANSFORM is a transform applied to account for differences in base frame coordinate conventions or differences in robot orientation. Our setup had both robots facing the same direction. A different setup may require a different transform. An important note is that when running the file in terminal, you should use the --keyboard-gripper argument. This allows you to open/close the gripper by pressing 'g,' or another custom keybind.

  `robotiq_gripper.py` includes necessary infrastructure to control the Hand-E gripper. There is no configuration needed assuming you are using a Hand-E with a UR5e.

  See data_collection README for more details on how to use these control files to collect episodes for fine-tuning.

  ## Limitations

  All of these files are made specifically for a setup involving a UR5e with a Robotiq Hand-E. Some may work on other setups, but not all. The main limitations of these files are that the gripper control file is Hand-E specific and the robot control file relies on UR RTDE. Some of the files offer built in 'stop' keybinds, but it is recommended to just use control-c in terminal instead.
