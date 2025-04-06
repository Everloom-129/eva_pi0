
import argparse
from tqdm import tqdm

from eva.runner import Runner
from eva.manager import load_runner


def collect_trajectory(runner: Runner, controller=None, n_traj=1, practice=False):
    if controller is not None:
        runner.set_controller(controller)
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
    parser.add_argument("--controller", default=None, choices=["occulus", "keyboard", "gello"])
    parser.add_argument("--practice", action="store_true")
    args = parser.parse_args()

    runner = load_runner()
    collect_trajectory(runner, controller=args.controller, n_traj=args.n_traj, practice=args.practice)
