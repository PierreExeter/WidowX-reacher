# widowx_arm_dev

ROS support for WidowX including MoveIt!, IKFast and Pick & Place / Sorting demo.

Optionally using SR300 for pointcloud.

Depending on interbotix/arbotix_ros/turtlebot2i [branch](https://github.com/Interbotix/arbotix_ros/tree/turtlebot2i) for gripper controller.

## Quick Start:

mkdir -p ~/widowx_arm/src

cd ~/widowx_arm/src

git clone https://github.com/Interbotix/widowx_arm.git .

git clone https://github.com/Interbotix/arbotix_ros.git -b turtlebot2i

cd ~/widowx_arm

catkin_make

source devel/setup.bash

roslaunch widowx_arm_bringup arm_moveit.launch sim:=false sr300:=false

## Object manipluation:

roslaunch widowx_arm_bringup arm_moveit.launch sim:=false sr300:=true

roslaunch widowx_block_manipulation block_sorting_demo.launch
