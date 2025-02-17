
import argparse
import threading
import numpy as np
import cv2

from franka_wliang.controllers.occulus import Occulus
from franka_wliang.env import FrankaEnv
from franka_wliang.runner import Runner
from franka_wliang.utils.misc_utils import run_threaded_command
from franka_wliang.manager import load_runner

def calibrate_camera(runner: Runner, camera_id):
    runner.display_camera_feed(camera_id=camera_id)
    runner.set_calibration_mode(camera_id)
    success = runner.calibrate_camera(camera_id, reset_robot=False)
    if success:
        print("Calibration complete!")
    else:
        print("Calibration failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera_id", type=str)
    args = parser.parse_args()

    runner = load_runner()
    calibrate_camera(runner, args.camera_id)
