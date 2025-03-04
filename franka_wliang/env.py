
import gym
import numpy as np
from copy import deepcopy

from franka_wliang.cameras.multi_camera_wrapper import MultiCameraWrapper
from franka_wliang.robot.server_interface import ServerInterface
from franka_wliang.utils.calibration_utils import load_calibration_info
from franka_wliang.utils.parameters import camera_type_dict, hand_camera_id, nuc_ip
from franka_wliang.utils.geometry_utils import change_pose_frame
from franka_wliang.utils.misc_utils import time_ms

class FrankaEnv(gym.Env):
    def __init__(self, action_space="cartesian_velocity", gripper_action_space=None, camera_kwargs={}, do_reset=True):
        # Initialize Gym Environment
        super().__init__()

        # Define Action Space #
        self.set_action_space(action_space)
        self.gripper_action_space = gripper_action_space

        # Robot Configuration
        self.reset_joints = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])
        # self.randomize_low = np.array([-0.1, -0.2, -0.1, -0.3, -0.3, -0.3])
        # self.randomize_high = np.array([0.1, 0.2, 0.1, 0.3, 0.3, 0.3])
        self.randomize_low = np.array([0, 0, 0, 0, 0, 0])
        self.randomize_high = np.array([0, 0, 0, 0, 0, 0])
        self.control_hz = 15

        if nuc_ip is None:
            from franka.robot import FrankaRobot

            self._robot = FrankaRobot()
        else:
            self._robot = ServerInterface(ip_address=nuc_ip)

        # Create Cameras
        self.camera_reader = MultiCameraWrapper(camera_kwargs)
        self.calibration_dict = load_calibration_info()
        self.camera_type_dict = camera_type_dict

        # Reset Robot
        if do_reset:
            self.reset()

    def step(self, action):
        # Check Action
        assert len(action) == self.DoF
        if self.check_action_range:
            assert (action.max() <= 1) and (action.min() >= -1)

        # Update Robot
        action_info = self.update_robot(
            action,
            action_space=self.action_space,
            gripper_action_space=self.gripper_action_space,
        )

        # Return Action Info
        return action_info

    def reset(self, randomize=False):
        self._robot.update_gripper(0, velocity=False, blocking=True)

        if randomize:
            noise = np.random.uniform(low=self.randomize_low, high=self.randomize_high)
        else:
            noise = None

        self._robot.update_joints(self.reset_joints, velocity=False, blocking=True, cartesian_noise=noise)

    def update_robot(self, action, action_space="cartesian_velocity", gripper_action_space=None, blocking=False):
        action_info = self._robot.update_command(
            action,
            action_space=action_space,
            gripper_action_space=gripper_action_space,
            blocking=blocking
        )
        return action_info

    def create_action_dict(self, action):
        return self._robot.create_action_dict(action)

    def read_cameras(self):
        return self.camera_reader.read_cameras()

    def get_state(self):
        read_start = time_ms()
        state_dict, timestamp_dict = self._robot.get_robot_state()
        timestamp_dict["read_start"] = read_start
        timestamp_dict["read_end"] = time_ms()
        return state_dict, timestamp_dict

    def get_camera_extrinsics(self, state_dict):
        # Adjust gripper camera by current pose
        extrinsics = deepcopy(self.calibration_dict)
        for cam_id in self.calibration_dict:
            if hand_camera_id not in cam_id:
                continue
            gripper_pose = state_dict["cartesian_position"]
            extrinsics[cam_id + "_gripper_offset"] = extrinsics[cam_id]
            extrinsics[cam_id] = change_pose_frame(extrinsics[cam_id], gripper_pose)
        return extrinsics

    def get_observation(self):
        obs_dict = {"timestamp": {}}

        # Robot State #
        state_dict, timestamp_dict = self.get_state()
        obs_dict["robot_state"] = state_dict
        obs_dict["timestamp"]["robot_state"] = timestamp_dict

        # Camera Readings #
        camera_obs, camera_timestamp = self.read_cameras()
        obs_dict.update(camera_obs)
        obs_dict["timestamp"]["cameras"] = camera_timestamp

        # Camera Info #
        obs_dict["camera_type"] = deepcopy(self.camera_type_dict)
        extrinsics = self.get_camera_extrinsics(state_dict)
        obs_dict["camera_extrinsics"] = extrinsics

        intrinsics = {}
        for cam in self.camera_reader.camera_dict.values():
            cam_intr_info = cam.get_intrinsics()
            for (full_cam_id, info) in cam_intr_info.items():
                intrinsics[full_cam_id] = info["cameraMatrix"]
        obs_dict["camera_intrinsics"] = intrinsics

        return obs_dict
    
    def set_action_space(self, action_space):
        print(f"Set action space to {action_space}")
        assert action_space in ["cartesian_position", "joint_position", "cartesian_velocity", "joint_velocity"]
        self.action_space = action_space
        self.check_action_range = "velocity" in action_space
        self.DoF = 7 if ("cartesian" in action_space) else 8
    
    def reload_calibration(self):
        self.calibration_dict = load_calibration_info()

    def close(self):
        self._robot.server.close()
