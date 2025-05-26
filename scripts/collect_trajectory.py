
import eva
from tqdm import tqdm

from eva.eva import init_context, init_parser
from eva.runner import Runner


def collect_trajectory(runner: Runner, n_traj=1):
    for _ in tqdm(range(n_traj), disable=(n_traj == 1)):
        runner.run_trajectory("collect")
        runner.print("Ready to reset, press any controller button...")
        while True:
            controller_info = runner.get_controller_info()
            if controller_info["success"] or controller_info["failure"]:
                break
        runner.reset_robot()


if __name__ == "__main__":
    parser = init_parser()
    parser.add_argument("-n", "--num_trajectories", type=int, default=1)
    args = parser.parse_args()

    with init_context(args) as runner:
        collect_trajectory(runner, n_traj=args.num_trajectories)
