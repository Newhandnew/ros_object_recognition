# ROS Object Recognition
---

## Goal
---
The goal of this project is to recognize object by 3D camera. Train an object by its name, save image as pattern and recognize objects by all saved patterns.


## Main Function
---
- get near objects: filter background by limiting the range of 3D camera, dilation, and get the largest detected area as object.(assume object is near)
- save object image as pattern
- use SURF algorithm to detect features
- compare multi objects by saved pattern
- implement as ROS service

## Framworks
---
- ROS
- ROS-openni2
- OpenCV3

## How to use
---
- install ROS
- install ros package openni2 to drive 3D camera
- install OpenCV3
- (in catkin workspace) catkin_make
- roslaunch object_recognition object_recognition.launch
- call learning pattern service by save_object (name)
- call recognizing objects by recognition (return object array)


