
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import threading

from franka_wliang.controllers.occulus import Occulus
from franka_wliang.env import FrankaEnv
from franka_wliang.runner import Runner
from franka_wliang.utils.misc_utils import run_threaded_command, keyboard_listener
from franka_wliang.manager import load_runner


# def collect_trajectory(runner: Runner, n_traj=1, practice=False):
#     runner.display_camera_feed()
#     with keyboard_listener() as keyboard:
#         for _ in tqdm(range(n_traj), disable=(n_traj == 1)):
#             runner.collect_trajectory(practice=practice)

#             print("Ready to reset, press any key or controller button...")
#             while True:
#                 controller_info = runner.get_controller_info()
#                 if controller_info["success"] or controller_info["failure"] or keyboard["pressed"] is not None:
#                     break
#             runner.reset_robot()

#     runner.close_camera_feed()


def collect_trajectory(runner: Runner, n_traj=1, practice=False):
    stop_camera_feed = threading.Event()
    def display_camera_feed():
        while not stop_camera_feed.is_set():
            try:
                camera_feed, cam_ids = runner.get_camera_feed()
            except:
                continue
            cols = [np.vstack(camera_feed[i:i+2]) for i in range(0, len(camera_feed), 2)]
            grid = np.hstack(cols)
            cv2.imshow("Camera Feed", cv2.cvtColor(cv2.resize(grid, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
    display_thread = run_threaded_command(display_camera_feed)

    with keyboard_listener() as keyboard:
        for _ in tqdm(range(n_traj), disable=(n_traj == 1)):
            runner.collect_trajectory(practice=practice)

            print("Ready to reset, press any key or controller button...")
            while True:
                controller_info = runner.get_controller_info()
                if controller_info["success"] or controller_info["failure"] or keyboard["pressed"] is not None:
                    break
            runner.reset_robot()
    
    stop_camera_feed.set()
    display_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_traj", type=int, default=1)
    parser.add_argument("--practice", action="store_true")
    parser.add_argument("--action_space", default="cartesian_velocity")
    args = parser.parse_args()

    runner = load_runner()
    runner.set_action_space(args.action_space)
    collect_trajectory(runner, n_traj=args.n_traj, practice=args.practice)
    runner.close_camera_feed()
