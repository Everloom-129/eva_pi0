from multiprocessing.managers import BaseManager
from collections import defaultdict

from eva.env import FrankaEnv
from eva.runner import Runner
from collections import defaultdict


class RunnerManager(BaseManager):
    pass


def init(controller, record_depth, record_pcd, post_process):
    camera_kwargs = defaultdict(
        lambda: {"depth": record_depth, "pointcloud": record_pcd}
    )
    env = FrankaEnv(camera_kwargs=camera_kwargs)
    runner = Runner(env=env, controller=controller, post_process=post_process)
    return runner


def start_runner(controller="occulus", record_depth=False, record_pcd=False, post_process=False):
    runner = init(controller, record_depth, record_pcd, post_process)
    RunnerManager.register("Runner", lambda: runner)
    manager = RunnerManager(address=("localhost", 50000), authkey=b"franka_runner")
    server = manager.get_server()
    print("Starting runner on localhost:50000...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down runner...")
        server.shutdown()
        server.server_close()
        runner.close()


def load_runner(manager=True, **kwargs):
    if not manager:
        runner = init(**kwargs)
        return runner

    RunnerManager.register("Runner")
    manager = RunnerManager(address=("localhost", 50000), authkey=b"franka_runner")
    try:
        manager.connect()
        return manager.Runner()
    except ConnectionRefusedError:
        print("ERROR: Failed to connect to runner server, please make sure start_runner.py is running.")
        raise
