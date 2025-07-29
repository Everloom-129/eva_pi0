
import zerorpc
import argparse
from contextlib import contextmanager


def init(args=None):
    # We import as late as possible to avoid overhead
    if args is None:
        parser = init_parser()
        args = parser.parse_args()

    try:
        runner = load_runner()
        runner.initialize(args.controller, args.controller_kwargs, args.disable_saving, args.disable_post_process, args.record_depth, args.record_pcd)
    except:
        try:
            env = load_env()
            env.initialize(args.action_space, args.gripper_action_space, args.camera_kwargs)
        except:
            from eva.env import FrankaEnv
            env = FrankaEnv()
        from eva.runner import Runner
        runner = Runner(env, controller=args.controller, controller_kwargs=args.controller_kwargs, disable_saving=args.disable_saving, disable_post_process=args.disable_post_process, record_depth=args.record_depth, record_pcd=args.record_pcd)
    return runner


@contextmanager
def init_context(args=None):
    runner = init(args)
    try:
        if isinstance(runner, zerorpc.Client):
            yield runner
        else:
            with runner:
                yield runner
    finally:
        runner.close()


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", default="oculus", choices=["oculus", "keyboard", "gello", "spacemouse", "policy", "proxy"])
    parser.add_argument("--controller_kwargs", type=dict, default={})
    parser.add_argument("--disable_saving", action="store_true", help="Disable saving data")
    parser.add_argument("--disable_post_process", action="store_true", help="Disable post processing")
    parser.add_argument("--record_depth", action="store_true", help="Record depth data")
    parser.add_argument("--record_pcd", action="store_true", help="Record point cloud data")
    return parser


def start_runner():
    from eva.runner import Runner
    from eva.env import FrankaEnv

    parser = init_parser()
    args = parser.parse_args()
    env = FrankaEnv()
    runner = Runner(env, **vars(args))
    server = zerorpc.Server(runner, heartbeat=None)
    server.bind("tcp://0.0.0.0:4545")
    print("Starting runner on tcp://0.0.0.0:4545...")
    try:
        server.run()
    except KeyboardInterrupt:
        print("Shutting down runner...")
        server.close()
        runner.close()

def load_runner():
    client = zerorpc.Client(heartbeat=None, timeout=None)
    client.connect("tcp://localhost:4545")
    return client

def start_env():
    from eva.env import FrankaEnv

    env = FrankaEnv()
    server = zerorpc.Server(env, heartbeat=None)
    server.bind("tcp://0.0.0.0:4646")
    print("Starting env on tcp://0.0.0.0:4646...")
    try:
        server.run()
    except KeyboardInterrupt:
        print("Shutting down env...")
        server.close()
        env.close()

def load_env():
    client = zerorpc.Client(heartbeat=None, timeout=None)
    client.connect("tcp://localhost:4646")
    return client
