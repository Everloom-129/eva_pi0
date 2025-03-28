
import argparse
from tqdm import tqdm

from franka_wliang.runner import Runner
from franka_wliang.manager import load_runner


def collect_trajectory(runner: Runner, n_traj=1, practice=False):
    for _ in tqdm(range(n_traj), disable=(n_traj == 1)):
        runner.run_trajectory(mode="collect")

        runner.print("Ready to reset, press any controller button...")
        while True:
            controller_info = runner.get_controller_info()
            if controller_info["success"] or controller_info["failure"]:
                break
        runner.reset_robot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_traj", type=int, default=1)
    parser.add_argument("--practice", action="store_true")
    parser.add_argument("--action_space", default="cartesian_velocity")
    args = parser.parse_args()

    runner = load_runner()
    runner.set_action_space(args.action_space)
    collect_trajectory(runner, n_traj=args.n_traj, practice=args.practice)
