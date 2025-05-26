
import eva
from eva.eva import init_context, init_parser
from eva.runner import Runner


def play_trajectory(runner: Runner, traj_path: str, action_space: str, autoplay=False, skip_reset=False):
    runner.set_controller("replayer", traj_path=traj_path, action_space=action_space)
    runner.run_trajectory("evaluate", wait_for_controller=not autoplay, reset_robot=not skip_reset)
    if not autoplay:
        runner.print("Ready to reset, press any controller button...")
        while True:
            controller_info = runner.get_controller_info()
            if controller_info["success"] or controller_info["failure"]:
                break
    runner.reset_robot()
    runner.set_prev_controller()


if __name__ == "__main__":
    parser = init_parser()
    parser.add_argument("-p", "--path", default="/home/franka/eva/trajectory.npy", type=str)
    parser.add_argument("--action_space", default="cartesian_position")
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--skip_reset", action="store_true")
    args = parser.parse_args()

    with init_context(args) as runner:
        runner.set_action_space(args.action_space)
        play_trajectory(runner, args.path, args.action_space, args.autoplay, args.skip_reset)
