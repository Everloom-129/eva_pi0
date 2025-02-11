
import numpy as np


class Replayer:
    def __init__(self, traj_path, action_space="cartesian_velocity"):  
        self.action_space = action_space
        self.phase = 0
        self.max_phase = 4
        self.threshold = 0.45
        self.name = "trajectory_replay"

        if traj_path.endswith(".npz"):
            if action_space == "cartesian_velocity":
                self.traj = np.load(traj_path)["actions_vel"]
            elif action_space == "cartesian_position":
                self.traj = np.load(traj_path)["actions_pos"]
        elif traj_path.endswith(".npy"):
            self.traj = np.load(traj_path)
        else:
            raise NotImplementedError("ERROR: Trajectory must be in npz or npy format!")
        
        self.traj_len = self.traj.shape[0]
        self.delay = 60
        self.t = 0

    def forward(self, observation, info=None):
        cur_ee_pos = np.zeros((7,))
        cur_ee_pos[:6] = observation["robot_state"]["cartesian_position"]
        if self.delay > 0:
            if self.action_space == "cartesian_velocity":
                action = np.zeros((7,))
            elif self.action_space == "cartesian_position":
                action = cur_ee_pos
            self.delay -= 1
        else:
            action = self.traj[self.t]
            self.t = min(self.t + 1, self.traj_len - 1)
        return action
