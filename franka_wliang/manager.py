
from multiprocessing.managers import BaseManager
from collections import defaultdict

from franka_wliang.controllers.occulus import Occulus
from franka_wliang.controllers.keyboard import Keyboard
from franka_wliang.env import FrankaEnv
from franka_wliang.runner import Runner
from collections import defaultdict


class RunnerManager(BaseManager):
    pass


def init(controller="occulus", control_mode="cartesian_velocity", record_depth=False, record_pcd=False, post_process=False):
    camera_kwargs = defaultdict(
        lambda: {"depth": record_depth, "pointcloud": record_pcd}
    )
    env = FrankaEnv(control_mode, camera_kwargs=camera_kwargs)
    if controller == "occulus":
        controller = Occulus()
    elif controller == "keyboard":
        controller = Keyboard()
    runner = Runner(env=env, controller=controller, post_process=post_process)
    return runner


def start_runner(controller="occulus", control_mode="cartesian_velocity", record_depth=False, record_pcd=False, post_process=False):
    runner = init(controller, control_mode, record_depth, record_pcd, post_process)
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
