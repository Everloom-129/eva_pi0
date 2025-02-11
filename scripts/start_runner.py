
import argparse

import franka_wliang
from franka_wliang.manager import start_runner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_mode", type=str, default="cartesian_velocity")
    args = parser.parse_args()

    start_runner(args.control_mode)