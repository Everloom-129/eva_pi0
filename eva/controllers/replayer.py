
import numpy as np

from eva.controllers.controller import Controller
from eva.utils.trajectory_utils import TrajectoryReader


class Replayer(Controller):
    def __init__(self, traj_path, action_space="cartesian_position", gripper_action_space="position"):
        self.action_space = action_space
        self.gripper_action_space = gripper_action_space
        self.gripper_action_space_hack = False
        if self.gripper_action_space == "position":
            self.gripper_action_space = "velocity"
            self.gripper_action_space_hack = True

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
        self._state = {
            "success": False,
            "failure": False,
            "movement_enabled": False,
            "controller_on": True,
        }
    
    def register_key(self, key):
        if key == ord(" "):
            self._state["movement_enabled"] = not self._state["movement_enabled"]
            print("Movement enabled:", self._state["movement_enabled"])
    
    def get_info(self):
        return self._state

    def forward(self, observation):
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

            if self.gripper_action_space_hack:
                # The Robotiq gripper will stop moving if it encounters a force limit (safety feature within polymetis)
                # In some scenarios, this will punish imprecise grasps by preventing the gripper from closing more once it has room
                # This happens less with gripper velocity control, so we convert gripper position to velocity
                gripper_action_gain = 3.0
                action[-1] = (action[-1] - observation["robot_state"]["gripper_position"]) * gripper_action_gain
                action[-1] = np.clip(action[-1], -1, 1)

            self.t = self.t + 1
            if self.t >= self.traj_len:
                self._state["success"] = True
        return action, {}

    def reset_state(self):
        self.t = 0
        self.delay = 0

    def close(self):
        pass