
import numpy as np

from eva.utils.trajectory_utils import TrajectoryReader


class Replayer:
    def __init__(self, traj_path, action_space="cartesian_velocity", gripper_action_space="velocity"):
        self.action_space = action_space
        self.gripper_action_space = gripper_action_space

        if traj_path.endswith(".npz"):
            if action_space == "cartesian_velocity":
                self.traj = np.load(traj_path)["actions_vel"]
            elif action_space == "cartesian_position":
                self.traj = np.load(traj_path)["actions_pos"]
        elif traj_path.endswith(".npy"):
            self.traj = np.load(traj_path)
        elif traj_path.endswith(".h5"):
            traj_reader = TrajectoryReader(traj_path, read_images=False)
            self.traj = []
            for i in range(traj_reader.length()):
                timestep = traj_reader.read_timestep()
                arm_action = timestep["action"][self.action_space]
                gripper_action = timestep["action"][self.gripper_action_space]
                if not timestep["observation"]["timestamp"]["skip_action"]:
                    self.traj.append(np.concatenate([arm_action, [gripper_action]]))
            self.traj = np.array(self.traj)
        else:
            raise ValueError(f"Invalid trajectory format: {traj_path}")
        
        self.traj_len = self.traj.shape[0]
        self.delay = 0
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
