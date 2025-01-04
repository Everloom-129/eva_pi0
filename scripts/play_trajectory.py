
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import threading

from franka_wliang.controllers.occulus import Occulus
from franka_wliang.env import FrankaEnv
from franka_wliang.runner import Runner
from franka_wliang.utils.misc_utils import run_threaded_command, keyboard_listener
import h5py

class PolicyWrapper:
    def __init__(
        self,
        traj_path: str,
    ):  
        self.phase = 0
        self.max_phase = 4
        self.threshold = 0.45
        self.deylay_period = 15
        self.name = "trajectory_replay"

        # self.traj = np.load("testaction.npy", allow_pickle=True).item()["actions"]
        if traj_path.endswith(".npz"):
            self.traj = np.load(traj_path)["actions_vel"] # shape (num_timesteps, 7)
            # self.traj = np.zeros((1000, 7)) # dummy action
        else:
            raise NotImplementedError("Only support npz files for now")
        self.traj_len = self.traj.shape[0]
        self.count = 0
        print("USING TRAJECTORY REPLAY POLICY WRAPPER!!!!!!!!!!!!!")

    def forward(self, observation, info=None):
        action = self.traj[min(self.count, self.traj_len - 1)]
        print("action: ", action)
        assert action.shape == (7,) # assume action is cartesian velocity
        self.count += 1
        return action

def play_trajectory(runner: Runner, traj_path: str):
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

    policy = PolicyWrapper(traj_path)
    with keyboard_listener() as keyboard:
        runner.play_trajectory(policy)
        print("Ready to reset, press any key or controller button...")
        while True:
            controller_info = runner.get_controller_info()
            if controller_info["success"] or controller_info["failure"] or keyboard["pressed"] is not None:
                break
        runner.reset_robot()
        print("done reset")
    
    stop_camera_feed.set()
    display_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_path", default="/home/franka/franka_wliang/data/success/2025-01-04/2025-01-04_16-57-08/trajectory.npz", type=str)
    args = parser.parse_args()

    env = FrankaEnv()
    controller = Occulus()
    runner = Runner(env=env, controller=controller)
    play_trajectory(runner, args.traj_path)


