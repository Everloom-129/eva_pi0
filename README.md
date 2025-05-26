
<div align="center">
  <img src="https://github.com/user-attachments/assets/1e36909c-62d8-4fd1-aa3d-333b98d5065e" width="480" />
</div>

Eva is an extendable, versatile, and adaptable robot infrastructure for the Franka Emika Panda, featuring:
- Modular design with atomic components, prioritizing flexibility and customizability.
- Lightweight and simple interfaces via terminal and live camera feed.
- Robust, fault-tolerant components supporting continuous operation.

This project is built on [DROID](https://github.com/droid-dataset/droid). Some components are completely revamped while others are lightly modified, but the hardware setup and data format remain unchanged.

## Installation
The DROID software and hardware setup form the foundation for Eva. Please install them via instructions [here](https://droid-dataset.github.io/droid/).

Afterwards, run the following:
```
git clone https://github.com/willjhliang/eva.git
cd eva
conda create -n eva python=3.10
conda activate eva
pip install -r requirements.txt
./sync_infra.sh
```

## Usage

Following the DROID setup, Eva runs on two machines:
- NUC: Handles low-level control of the Franka Emika with a server built on [Polymetis](https://facebookresearch.github.io/fairo/polymetis/).
- Laptop: Handles high-level logic (policy inference, teleoperation, etc) with a runner that executes user scripts.

We recommend the following tmux setup:
```
+-------------------------+-------------------------+
|                         |                         |
|      Server (NUC)       |     Runner (Laptop)     |
+-------------------------+                         |
|    Scripts (Laptop)     |                         |
|                         |                         |
+-------------------------+-------------------------+
```

### Startup

1. On the NUC, run
```bash
cd eva/eva/robot
./launch_server.sh
```
2. On the laptop, run
```bash
conda activate eva
cd eva/scripts
python start_runner.py
```

### Scripts

After the server and runner are started, you can execute scripts found in `eva/scripts/`. Some of the main functions include:
- `collect_trajectory.py`: Collects teleoperated trajectories saved in `eva/data/`.
- `play_trajectory.py`: Replays a selected trajectory.
- `process_trajectory.py`: Processes the compressed trajectory data into a more usable format.
- `calibrate_camera.py`: Calibrates a camera using the Charuco board.
- `check_calibration.py`: Overlays a gripper annotation on the camera feed.
- `take_pictures.py`: Saves camera pictures to `eva/data/images`.
- `reset_robot.py`: Resets the robot pose to default.

Each script has its own specific arguments as well as a set of general-purpose arguments (found in `eva.py`). Some crucial ones are:
- `--controller`: Sets the control method for the robot, such as teleoperation controllers (e.g., Occulus) or learned policies.
- `--disable_post_process`: Disables online trajectory post-processing, saving space and freeing compute.
- `--record_depth` and `--record_pcd`: Records additional depth and point cloud observations besides the standard RGB.

### Development

Code development should be entirely done on the laptop, and to sync the codebase with the NUC, run `./sync_infra.sh`. Remember to restart the NUC server or runner if code changes affect them.

If you are using Eva and plan to make significant changes, **please work in a copy of this directory** (e.g., `eva_wliang`).
