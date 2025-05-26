
import eva
import argparse
import threading
import numpy as np
import cv2

from eva.eva import init_context, init_parser
from eva.runner import Runner
from eva.utils.misc_utils import run_threaded_command

def calibrate_camera(runner: Runner, camera_id, advanced_calibration=False):
    if advanced_calibration:
        runner.enable_advanced_calibration()
        print("WARNING: This is an experimental feature that isn't fully tested, use at your own risk!")
        print("It will save the high-resolution intrinsics to calibration.json, if you intend to use standard resolution later, please edit intrinsics in calibration.json manually!")
    runner.set_calibration_mode(camera_id)
    success = runner.calibrate_camera(camera_id, reset_robot=False)
    if success:
        print("Calibration complete!")
    else:
        print("Calibration failed")
    if advanced_calibration:
        runner.disable_advanced_calibration()


if __name__ == "__main__":
    parser = init_parser()
    parser.add_argument("-c", "--camera_id", type=str)
    parser.add_argument("--advanced", action="store_true")
    args = parser.parse_args()

    with init_context(args) as runner:
        calibrate_camera(runner, args.camera_id, args.advanced)
