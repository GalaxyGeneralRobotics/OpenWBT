<div align="center">
  <h1 align="center"> OpenWBT </h1>
  <h3 align="center"> GALBOT · Tsinghua </h3>
<!--   <p align="center">
    <a href="README.md"> English </a> | <a href="README_zh.md">中文</a>
  </p>     -->

[中文](README_zh.md) |[English](README.md)

</div>

> ⚠️ This project is based on a development-stage version of Isaac Sim, and APIs may change frequently.


# Introduction
The technical implementation of this project is mainly supported by [R2S2](https://zzk273.github.io/R2S2/). This repository implements **whole body teleoperation** of the Unitree G1 and H1 humanoid robot using Apple Vision Pro. The system supports **both real robots and simulation environments**, enabling users to control the robot to perform movements such as walking, squatting, bending, grasping, and lifting, thereby significantly expanding the robot's operational capabilities and enabling it to accomplish a wider range of tasks.

## Demonstration:
TODO:

<p align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <a href="https://www.bilibili.com/video/x" target="_blank">
          <img src="./img/x.jpg" alt="real.gif" width="75%">
        </a>
        <p><b> real demonstration</b></p>
      </td>
      <td align="center" width="50%">
        <a href="https://www.bilibili.com/video/x" target="_blank">
          <img src="./img/x.jpg" alt="sim.gif" width="75%">
        </a>
        <p><b> sim demonstration </b></p>
      </td>
    </tr>
  </table>
</p>

# Hardware List and Environment Configuration
see [Installation.md](installation.md) 

# Usage
## Starting the Robot

First turn on the robot:
- Press and hold the power button briefly, then hold it again until the blue light on the head turns on and stops blinking.

Then use the unitree remote controller to enter the damping mode (you can alse refer to [Unitree official document](https://support.unitree.com/home/zh/G1_developer/remote_control)):
- Press and hold the power button briefly, then hold it again.
- Press L2+R2 to enter deployable mode.
- Press L2+A, the robot will show a ready pose.
- Press L2+B to enter damping mode.

## Connecting to the Robot via Ethernet and Starting the Image Service

The first time you connect to the robot, you need to configure the `wifi` to install some dependencies. You can refer to [FAQ from unitree](https://support.unitree.com/home/zh/G1_developer/FAQ) for wifi configuration. We provide an example approach:

Modify the `wifi` name and corresponding password in the `01-netcfg.yaml` file, and place the `01-netcfg.yaml` file in `/etc/netplan/` on the robot:

```bash
scp ./01-netcfg.yaml unitree@192.168.123.164:~/ # Transfer to the robot's root directory

# Access the remote desktop
ssh unitree@192.168.123.164
123 # password
1 # choose ros2 foxy(1)
sudo cp 01-netcfg.yaml /etc/netplan/
netplan apply # *You may be prompted to restart the network service during this process. Try a few more times if necessary.*

# Install dependencies
pip install pyzmq
pip install pyrealsense2
```

Transfer the Python file for starting the image service to the robot and run it

```bash
scp ./deploy/teleop/image_server/image_server.py unitree@192.168.123.164:~/image_server/
scp unitree@192.168.123.164:~/image_server/image_server.py  ./deploy/teleop/

ssh unitree@192.168.123.164
123 # password
1 # choose ros2 foxy(1)
cd image_server
python image_server.py
```
## Executing the Teleoperation Program
Connect the computer to the joysticks and grant access permissions to both hands. By default, the left and right joysticks are located at /dev/ttyACM0 and /dev/ttyACM1, respectively.

```bash
sudo chmod -R 777 /dev/ttyACM0; sudo chmod -R 777 /dev/ttyACM1
```

Running in the Real:
``` bash
source /opt/ros/foxy/setup.sh; source ~/unitree_ros2/setup.sh
python -m deploy.run_teleoperation_real --config run_teleoperation.yaml --net eno1
```
Running in the Simulator:
```bash
source /opt/ros/foxy/setup.sh; source ~/unitree_ros2/setup.sh
python -m deploy.run_teleoperation_mujoco --config run_teleoperation.yaml
```
Note:

- Replace eno1 with the correct network interface name.
- To enter debug mode (action from policy will not be executed) for real robot teleoperation, run:
```bash
source /opt/ros/foxy/setup.sh; source ~/unitree_ros2/setup.sh
python -m deploy.run_teleoperation_real --config run_teleoperation.yaml --net eno1 --debug
```

## Entering VR

*Prerequisites: The teleoperation program has been executed on the user's computer, and the robot's image service has been enabled.​*

**For the first-time connection and use in this project, please refer to the corresponding section in `installation.md`.​**

Open the browser on Apple Vision Pro and enter `https://192.168.123.2:8012?ws=wss://192.168.123.2:8012`.
Click "Enter VR" and allow tracking to view the first-person perspective. At this point, you can use the Vision Pro and joystick to control the actual robot.​

Note: 
- Replace the 192.168.123.2 with your own wifi IP address.

If images fail to display in VR, please double-check the following:
- You have followed the steps in installation.md, including sending, installing, and trusting the certificate
- The URL is entered exactly, with no typos or missing characters

## Control Robot Using Joystick


1. Press the `power` button on the right joystick briefly to make the robot enter the ready position.
2. Press and hold the `power` button on the right joystick for 1 seconds to make the robot enter startup mode.
3. Press the `A` button on the left joystick to make the robot enter locomotion mode - use the joysticks on both the left and right joysticks to control movement and turning.
4. In locomotion mode, press the `D` key on the left joystick to stop the robot from walking.
5. Press the `A` button on the right joystick to make the robot enter body-pose-adjustment mode - use the joysticks on both the left and right joysticks to control squatting, standing, and upper body pitch angles.
- *Press the `D` button on the right joystick to make the robot enter safe damping mode.*


# TODO List

- \[x\] Release the R2S2 paper with demos.
- \[x\] Release Unitree G1 whoble body teleoperation in real world.
- \[x\] Release Unitree G1 whoble body teleoperation in mujoco.
- \[ \] Release Unitree H1 whoble body teleoperation.


# Citation

If you find our work helpful, please cite:

```bibtex
@article{zhang2025unleashing,
  title={Unleashing Humanoid Reaching Potential via Real-world-Ready Skill Space},
  author={Zhang, Zhikai and Chen, Chao and Xue, Han and Wang, Jilong and Liang, Sikai and Liu, Yun and Zhang, Zongzhang and Wang, He and Yi, Li},
  journal={arXiv preprint arXiv:2505.10918},
  year={2025}
}
```

# License

All code of OpenWBT is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License </a><a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>. It is strictly forbidden to use it for commercial purposes before asking our team.

# Acknowledgments
This code is built upon the following open-source repositories. Please visit the respective links to view their licenses:

1. https://github.com/OpenTeleVision/TeleVision
2. https://github.com/unitreerobotics/avp_teleoperate

# Contact us 

If you'd like to discuss anything, feel free to send an email to zhikaizhang273@gmail.com or add WeChat: zzk273939.