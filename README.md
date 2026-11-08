# Vision-Guided Pick-and-Place with Franka Emika Panda

ROS2 Humble pipeline for perception-driven pick-and-place on a simulated Panda arm.
MoveIt2 handles planning in RViz (fake hardware); PyBullet provides the simulated
RGB-D camera and objects. YOLOv8 detection and automated trials are in progress.

## Status
- [x] MoveIt2 motion node: pick, lift, place, retreat (pymoveit2)
- [x] Table collision object in the planning scene
- [ ] PyBullet scene and RGB-D camera
- [ ] YOLOv8 detection and depth deprojection
- [ ] Orchestrator, per-trial CSV logging, evaluation

## Setup
- Ubuntu 22.04 (WSL2), ROS2 Humble, `ros-humble-moveit`,
  `ros-humble-moveit-resources-panda-moveit-config`, `ros-humble-ros2-controllers`
- Clone [pymoveit2](https://github.com/AndrejOrsula/pymoveit2) into `src/` next to this package
- `colcon build --symlink-install`

## Run
Terminal 1: `ros2 launch moveit_resources_panda_moveit_config demo.launch.py`
Terminal 2: `ros2 run pick_place motion_node`

