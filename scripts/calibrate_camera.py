
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
    stop_camera_feed = threading.Event()
    def display_camera_feed():
        while not stop_camera_feed.is_set():
            try:
                camera_feed, cam_ids = runner.get_camera_feed()
                camera_feed = [feed for i, feed in enumerate(camera_feed) if str(camera_id) in cam_ids[i] ]
            except:
                continue
            cols = [np.vstack(camera_feed[i:i+2]) for i in range(0, len(camera_feed), 2)]
            grid = np.hstack(cols)
            cv2.imshow("Camera Feed", cv2.cvtColor(cv2.resize(grid, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
    display_thread = run_threaded_command(display_camera_feed)

    runner.set_calibration_mode(camera_id)
    success = runner.calibrate_camera(camera_id, reset_robot=False)
    if success:
        print("Calibration complete!")
    else:
        print("Calibration failed")

    stop_camera_feed.set()
    display_thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera_id", type=str)
    args = parser.parse_args()

    runner = load_runner()
    calibrate_camera(runner, args.camera_id)
