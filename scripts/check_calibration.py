
import argparse
import threading
import numpy as np
import cv2

from franka_wliang.controllers.occulus import Occulus
from franka_wliang.env import FrankaEnv
from franka_wliang.runner import Runner
from franka_wliang.utils.misc_utils import run_threaded_command
from franka_wliang.manager import load_runner


def check_calibration(runner: Runner):
    print("Annotating end effector pose in camera feed...")
    runner.reload_calibration()
    runner.check_calibration()


if __name__ == "__main__":
    runner = load_runner()
    check_calibration(runner)
