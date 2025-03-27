import time
import zmq
import numpy as np
from collections.abc import Callable

from franka_wliang.utils.misc_utils import run_threaded_command


def vec_to_reorder_mat(vec):
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X



class GELLODevice():
    def __init__(self):
        super().__init__()
        # Set up ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://127.0.0.1:5555")  # Adjust the address if needed
        self.callbacks = {}

    def reset(self):
        pass  # Implement reset logic if necessary

    def add_callback(self, key: str, func: Callable):
        self.callbacks[key] = func

    def advance(self):
        # Send request to zmq server
        self.socket.send(b"get_joint_state")
        message = self.socket.recv()
        gello_action = np.frombuffer(message, dtype=np.float32)
        return gello_action
        # # Convert to torch tensor

        # action = np.zeros(8)
        # action[:8] = gello_action
        # action[-1] = gello_action[-1]  # Copy other gripper finger
        # # actions = torch.tensor(action) - joint_positions
        # actions = torch.tensor(action)
        # return actions


class Gello:
    def __init__(
        self,
        right_controller: bool = True,
        max_lin_vel: float = 1,
        max_rot_vel: float = 1,
        max_gripper_vel: float = 1,
        spatial_coeff: float = 1,
        pos_action_gain: float = 5,
        rot_action_gain: float = 2,
        gripper_action_gain: float = 3,
        rmat_reorder: list = [-2, -1, -3, 4],
    ):
        self.action_space = "joint_position"
        self.gello_device = GELLODevice()
        self.vr_to_global_mat = np.eye(4)
        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.max_gripper_vel = max_gripper_vel
        self.spatial_coeff = spatial_coeff
        self.pos_action_gain = pos_action_gain
        self.rot_action_gain = rot_action_gain
        self.gripper_action_gain = gripper_action_gain
        self.global_to_env_mat = vec_to_reorder_mat(rmat_reorder)
        self.controller_id = "r" if right_controller else "l"
        self.reset_orientation = True
        self.reset_state()

        # Start State Listening Thread #
        self.running = True
        run_threaded_command(self._update_internal_state)

    def reset_state(self):
        self._state = {
            "poses": {},
            "buttons": {"A": False, "B": False},
            "movement_enabled": False,
            "controller_on": True,
        }
        self.update_sensor = True
        self.reset_origin = True
        self.robot_origin = None
        self.vr_origin = None
        self.vr_state = None

    def _update_internal_state(self, num_wait_sec=5, hz=50):
        last_read_time = time.time()
        while self.running:
            # Regulate Read Frequency #
            time.sleep(1 / hz)

            # Read Controller
            time_since_read = time.time() - last_read_time
            
            # TODO: Get joint poses from GELLO instead here
            # poses, buttons = self.oculus_reader.get_transformations_and_buttons()
            gello_state = self.gello_device.advance()
            gello_joints = gello_state[:-1]
            gello_gripper = gello_state[-1]


            # Temporarily always enable action
            movement_enabled = True

            self._state["controller_on"] = time_since_read < num_wait_sec
            # if poses == {}:
            #     continue

            # Determine Control Pipeline #
            toggled = self._state["movement_enabled"] != movement_enabled
            self.update_sensor = self.update_sensor or movement_enabled
            # self.reset_orientation = self.reset_orientation or buttons["RJ"]
            self.reset_orientation = True
            self.reset_origin = self.reset_origin or toggled

            # Save Info #
            # TODO: Save GELLO info here instead
            self._state["joints"] = gello_joints
            self._state["gripper"] = gello_gripper


            self._state["movement_enabled"] = movement_enabled
            self._state["controller_on"] = True
            last_read_time = time.time()

            # Update Definition Of "Forward" #
            # TODO: Not sure what this does
            # stop_updating = self._state["buttons"]["RJ"] or self._state["movement_enabled"]
            # stop_updating = False
            # if self.reset_orientation:
            #     rot_mat = np.asarray(self._state["poses"][self.controller_id])
            #     if stop_updating:
            #         self.reset_orientation = False
            #     # try to invert the rotation matrix, if not possible, then just use the identity matrix                
            #     try:
            #         rot_mat = np.linalg.inv(rot_mat)
            #     except:
            #         print(f"exception for rot mat: {rot_mat}")
            #         rot_mat = np.eye(4)
            #         self.reset_orientation = True
            #     self.vr_to_global_mat = rot_mat

    def _process_reading(self):
        # rot_mat = np.asarray(self._state["poses"][self.controller_id])
        # rot_mat = self.global_to_env_mat @ self.vr_to_global_mat @ rot_mat
        # vr_pos = self.spatial_coeff * rot_mat[:3, 3]
        # vr_quat = rmat_to_quat(rot_mat[:3, :3])
        # vr_gripper = self._state["buttons"]["rightTrig"][0]

        # self.vr_state = {"pos": vr_pos, "quat": vr_quat, "gripper": vr_gripper}

        # TODO: Process and save to self.gello_state instead
        self.gello_state = {"joints": self._state["joints"], "gripper": self._state["gripper"]}

    def _limit_velocity(self, lin_vel, rot_vel, gripper_vel):
        """Scales down the linear and angular magnitudes of the action"""
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        gripper_vel_norm = np.linalg.norm(gripper_vel)
        if lin_vel_norm > self.max_lin_vel:
            lin_vel = lin_vel * self.max_lin_vel / lin_vel_norm
        if rot_vel_norm > self.max_rot_vel:
            rot_vel = rot_vel * self.max_rot_vel / rot_vel_norm
        if gripper_vel_norm > self.max_gripper_vel:
            gripper_vel = gripper_vel * self.max_gripper_vel / gripper_vel_norm
        return lin_vel, rot_vel, gripper_vel

    def _calculate_action(self, state_dict, include_info=False):
        # Read Sensor #
        if self.update_sensor:
            self._process_reading()
            self.update_sensor = False

        # Read Observation


        # robot_pos = np.array(state_dict["cartesian_position"][:3])
        # robot_euler = state_dict["cartesian_position"][3:]
        # robot_quat = euler_to_quat(robot_euler)
        # robot_gripper = state_dict["gripper_position"]

        # robot_joints = np.array(state_dict["joint_positions"])
        # robot_gripper = state_dict["gripper_position"]



        # TODO: I think this should be removed
        # Reset Origin On Release #
        # if self.reset_origin:
        #     self.robot_origin = {"pos": robot_pos, "quat": robot_quat}
        #     self.vr_origin = {"pos": self.vr_state["pos"], "quat": self.vr_state["quat"]}
        #     self.reset_origin = False

      
        #Find the difference between each of the joints

        # robot_joint_offset = robot_joints - self.gello_state["joints"]
        # gripper_offset = robot_gripper - self.gello_state["gripper"]


        # Old End effector control code
        # # Calculate Positional Action #
        # robot_pos_offset = robot_pos - self.robot_origin["pos"]
        # target_pos_offset = self.vr_state["pos"] - self.vr_origin["pos"]
        # pos_action = target_pos_offset - robot_pos_offset
        # # Calculate Euler Action #
        # robot_quat_offset = quat_diff(robot_quat, self.robot_origin["quat"])
        # target_quat_offset = quat_diff(self.vr_state["quat"], self.vr_origin["quat"])
        # quat_action = quat_diff(target_quat_offset, robot_quat_offset)
        # euler_action = quat_to_euler(quat_action)

        # # Calculate Gripper Action #
        # gripper_action = self.vr_state["gripper"] - robot_gripper

        # # Calculate Desired Pose #
        # target_pos = pos_action + robot_pos
        # target_euler = add_angles(euler_action, robot_euler)
        # target_cartesian = np.concatenate([target_pos, target_euler])
        # target_gripper = self.vr_state["gripper"]

        # # Scale Appropriately #
        # pos_action *= self.pos_action_gain
        # euler_action *= self.rot_action_gain
        # gripper_action *= self.gripper_action_gain

        # TODO: Fix this
        # lin_vel, rot_vel, gripper_vel = self._limit_velocity(pos_action, euler_action, gripper_action)

        # Prepare Return Values #
        info_dict = {"target_joint_positions": self.gello_state["joints"], "target_gripper_position": self.gello_state["gripper"]}
        # action = np.concatenate([robot_joint_offset, [gripper_offset]])
        # action = action.clip(-1, 1)
        action = np.concatenate([self.gello_state["joints"], [self.gello_state["gripper"]]])

        # Return #
        if include_info:
            return action, info_dict
        else:
            return action

    def get_info(self):
        return {
            "success": self._state["buttons"]["A"],
            "failure": self._state["buttons"]["B"],
            "movement_enabled": self._state["movement_enabled"],
            "controller_on": self._state["controller_on"],
        }

    def forward(self, obs_dict, include_info=False):
        # if self._state["poses"] == {}:
        #     action = np.zeros(7)
        #     if include_info:
        #         return action, {}
        #     else:
        #         return action
        return self._calculate_action(obs_dict["robot_state"], include_info=include_info)
    
    def register_key(self, key):
        if key == ord(" "):
            self.reset_origin = True
    
    def close(self):
        self.running = False
        self.oculus_reader.stop()
