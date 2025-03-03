
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import threading

from franka_wliang.controllers.replayer import Replayer
from franka_wliang.env import FrankaEnv
from franka_wliang.runner import Runner
from franka_wliang.utils.misc_utils import run_threaded_command, keyboard_listener
from franka_wliang.manager import load_runner


def play_trajectory(runner: Runner, traj_path: str, action_space: str, autoplay=False, skip_reset=False):
    policy = Replayer(traj_path, action_space)
    with keyboard_listener() as keyboard:
        runner.play_trajectory(policy, wait_for_controller=not autoplay, reset_robot=not skip_reset)
        if not autoplay:
            print("Ready to reset, press any key or controller button...")
            while True:
                controller_info = runner.get_controller_info()
                if controller_info["success"] or controller_info["failure"] or keyboard["pressed"] is not None:
                    break
        runner.reset_robot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_path", default="/home/franka/franka_wliang/trajectory_final.npy", type=str)
    parser.add_argument("--action_space", default="cartesian_velocity")
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--skip_reset", action="store_true")
    args = parser.parse_args()

    runner = load_runner()
    runner.set_action_space(args.action_space)
    play_trajectory(runner, args.traj_path, args.action_space, args.autoplay, args.skip_reset)
