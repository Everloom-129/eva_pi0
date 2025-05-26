
import eva
import argparse
import threading
import numpy as np
import cv2

from eva.eva import init_context, init_parser
from eva.runner import Runner


def check_calibration(runner: Runner):
    print("Annotating end effector pose in camera feed...")
    runner.set_controller("occulus")
    runner.reload_calibration()
    runner.check_calibration()
    runner.set_prev_controller()


if __name__ == "__main__":
    with init_context() as runner:
        check_calibration(runner)
