import os
import time
from copy import deepcopy
from datetime import datetime
import cv2
import h5py
from pathlib import Path
import shutil
import threading
import numpy as np

from franka_wliang.utils.trajectory_utils import collect_trajectory
from franka_wliang.utils.calibration_utils import calibrate_camera, check_calibration_info, save_calibration_info
from franka_wliang.utils.misc_utils import data_dir, run_threaded_command
from franka_wliang.utils.parameters import hand_camera_id, code_version, robot_serial_number, robot_type


class Runner:
    def __init__(self, env, controller, policy=None, save_data=True, save_traj_dir=None):
        self.env = env
        self.controller = controller
        self.policy = policy

        self.last_traj_path = None
        self.traj_running = False
        self.obs_pointer = {}

        # Get Camera Info #
        self.cam_ids = list(env.camera_reader.camera_dict.keys())
        self.cam_ids.sort()

        _, full_cam_ids = self.get_camera_feed()
        self.num_cameras = len(full_cam_ids)
        self.full_cam_ids = full_cam_ids
        self.advanced_calibration = False

        self.stop_camera_feed = None
        self.display_thread = None

        # Make Sure Log Directorys Exist #
        if save_traj_dir is None:
            save_traj_dir = data_dir
        self.success_logdir = os.path.join(save_traj_dir, "success", datetime.now().strftime("%Y-%m-%d"))
        self.failure_logdir = os.path.join(save_traj_dir, "failure", datetime.now().strftime("%Y-%m-%d"))
        self.eval_logdir = os.path.join(save_traj_dir, "eval", datetime.now().strftime("%Y-%m-%d"))
        if not os.path.isdir(self.success_logdir):
            os.makedirs(self.success_logdir)
        if not os.path.isdir(self.failure_logdir):
            os.makedirs(self.failure_logdir)
        self.save_data = save_data

    def reset_robot(self, randomize=False):
        self.env._robot.establish_connection()
        self.controller.reset_state()
        self.env.reset(randomize=randomize)

    def get_controller_info(self):
        info = self.controller.get_info()
        return deepcopy(info)

    def enable_advanced_calibration(self):
        self.advanced_calibration = True
        self.env.camera_reader.enable_advanced_calibration()

    def disable_advanced_calibration(self):
        self.advanced_calibration = False
        self.env.camera_reader.disable_advanced_calibration()

    def set_calibration_mode(self, cam_id):
        self.env.camera_reader.set_calibration_mode(cam_id)

    def set_trajectory_mode(self):
        self.env.camera_reader.set_trajectory_mode()

    def collect_trajectory(self, info=None, practice=False, reset_robot=True):
        traj_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        info = {} if info is None else info
        info["time"] = traj_name
        info["robot_serial_number"] = f"{robot_type}-{robot_serial_number}"
        info["version_number"] = code_version

        if practice or not self.save_data:
            save_filepath = None
            recording_dir = None
        else:
            if len(self.full_cam_ids) != 6:
                raise ValueError("WARNING: User is trying to collect data without all three cameras running!")
            
            save_dir = os.path.join(self.failure_logdir, traj_name)  # Assume failure first, move to success post-run
            recording_dir = os.path.join(save_dir, "recordings")
            save_filepath = os.path.join(save_dir, "trajectory.h5")
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(recording_dir, exist_ok=True)
            save_calibration_info(os.path.join(self.failure_logdir, info["time"], "calibration.json"))

        # Collect Trajectory #
        self.traj_running = True
        self.env._robot.establish_connection()
        controller_info = collect_trajectory(
            self.env,
            controller=self.controller,
            metadata=info,
            policy=self.policy,
            obs_pointer=self.obs_pointer,
            reset_robot=reset_robot,
            recording_folderpath=recording_dir,
            save_filepath=save_filepath,
            wait_for_controller=True,
        )
        self.traj_running = False
        self.obs_pointer = {}

        if save_filepath is not None:
            if controller_info["success"]:
                new_save_dir = os.path.join(self.success_logdir, traj_name)
                shutil.move(save_dir, new_save_dir)
                save_dir = new_save_dir
        
        self.last_traj_name = traj_name
        self.last_traj_path = save_dir

    def play_trajectory(self, policy_wrapper, reset_robot=True):
        '''
        Assume traj_path is a directory containing a trajectory.h5 file.
        '''

        traj_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        info = {}
        info["time"] = traj_name
        info["robot_serial_number"] = f"{robot_type}-{robot_serial_number}"
        info["version_number"] = code_version

        save_dir = os.path.join(self.eval_logdir, traj_name)  # Assume failure first, move to success post-run
        recording_dir = os.path.join(save_dir, "recordings")
        save_filepath = os.path.join(save_dir, "trajectory.h5")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(recording_dir, exist_ok=True)
        save_calibration_info(os.path.join(self.eval_logdir, info["time"], "calibration.json"))

        self.traj_running = True
        self.env._robot.establish_connection()
        controller_info = collect_trajectory(
            self.env,
            controller=self.controller,
            metadata=info,
            policy=policy_wrapper,
            horizon=policy_wrapper.traj_len - 1,
            obs_pointer=self.obs_pointer,
            reset_robot=reset_robot,
            recording_folderpath=recording_dir,
            save_filepath=save_filepath,
            wait_for_controller=True,
        )
        self.traj_running = False
        self.obs_pointer = {}

        # if save_filepath is not None:
        #     if controller_info["success"]:
        #         new_save_dir = os.path.join(self.eval_logdir, traj_name)
        #         shutil.move(save_dir, new_save_dir)
        #         save_dir = new_save_dir
        
        self.last_traj_name = traj_name
        self.last_traj_path = save_dir

    def calibrate_camera(self, cam_id, reset_robot=True):
        self.traj_running = True
        self.env._robot.establish_connection()
        success = calibrate_camera(
            self.env,
            cam_id,
            controller=self.controller,
            obs_pointer=self.obs_pointer,
            wait_for_controller=True,
            reset_robot=reset_robot,
        )
        self.traj_running = False
        self.obs_pointer = {}
        return success

    def check_calibration_info(self, remove_hand_camera=False):
        info_dict = check_calibration_info(self.full_cam_ids)
        if remove_hand_camera:
            info_dict["old"] = [cam_id for cam_id in info_dict["old"] if (hand_camera_id not in cam_id)]
        return info_dict

    def get_gui_imgs(self, obs):
        all_cam_ids = list(obs["image"].keys())
        all_cam_ids.sort()

        gui_images = []
        for cam_id in all_cam_ids:
            img = cv2.cvtColor(obs["image"][cam_id], cv2.COLOR_BGRA2RGB)
            gui_images.append(img)
        
        # depth_cam_ids = list(obs["depth"].keys())
        # depth_cam_ids.sort()
        # import numpy as np
        # for cam_id in depth_cam_ids:
        #     depth = np.nan_to_num(obs["depth"][cam_id])
        #     img = cv2.cvtColor(depth, cv2.COLOR_BGRA2RGB)
        #     gui_images.append(img)
        # all_cam_ids.extend([id+"_depth" for id in depth_cam_ids])

        return gui_images, all_cam_ids

    def get_camera_feed(self):
        if self.traj_running:
            if "image" not in self.obs_pointer:
                raise ValueError
            obs = deepcopy(self.obs_pointer)
        else:
            obs = self.env.read_cameras()[0]
        gui_images, cam_ids = self.get_gui_imgs(obs)
        return gui_images, cam_ids
    
    def get_obs(self):
        if self.traj_running:
            if "image" not in self.obs_pointer:
                raise ValueError
            obs = deepcopy(self.obs_pointer)
        else:
            obs = self.env.read_cameras()[0]
        return obs
    
    def get_state(self):
        return self.env.get_observation()

    def display_camera_feed(self, camera_id=None):
        self.stop_camera_feed = threading.Event()
        def display_thread():
            while not self.stop_camera_feed.is_set():
                try:
                    camera_feed, cam_ids = self.get_camera_feed()
                    if camera_id is not None:
                        camera_feed = [feed for i, feed in enumerate(camera_feed) if str(camera_id) in cam_ids[i] ]
                except:
                    continue
                cols = [np.vstack(camera_feed[i:i+2]) for i in range(0, len(camera_feed), 2)]
                grid = np.hstack(cols)
                cv2.imshow("Camera Feed", cv2.cvtColor(cv2.resize(grid, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()
        self.display_thread = run_threaded_command(display_thread)
        
    # def display_camera_feed(self, camera_id=None):
    #     self.stop_camera_feed = threading.Event()
    #     def display_thread():
    #         while not self.stop_camera_feed.is_set():
    #             try:
    #                 camera_feed, cam_ids = self.get_camera_feed()
    #             except:
    #                 continue
    #             cols = [np.vstack(camera_feed[i:i+2]) for i in range(0, len(camera_feed), 2)]
    #             grid = np.hstack(cols)
    #             cv2.imshow("Camera Feed", cv2.cvtColor(cv2.resize(grid, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR))
    #             if cv2.waitKey(1) & 0xFF == ord("q"):
    #                 break
    #         cv2.destroyAllWindows()
    #     self.display_thread = run_threaded_command(display_thread)

    def close_camera_feed(self):
        print("Closing camera feed...")
        if self.stop_camera_feed is not None and self.display_thread is not None:
            self.stop_camera_feed.set()
            self.display_thread.join()
            self.stop_camera_feed = None
            self.display_thread = None
            print("Camera feed closed.")

    def change_trajectory_status(self, success=False):
        if (self.last_traj_path is None) or (success == self.traj_saved):
            return

        save_filepath = os.path.join(self.last_traj_path, "trajectory.h5")
        traj_file = h5py.File(save_filepath, "r+")
        traj_file.attrs["success"] = success
        traj_file.attrs["failure"] = not success
        traj_file.close()

        if success:
            new_traj_path = os.path.join(self.success_logdir, self.last_traj_name)
            os.rename(self.last_traj_path, new_traj_path)
            self.last_traj_path = new_traj_path
            self.traj_saved = True
        else:
            new_traj_path = os.path.join(self.failure_logdir, self.last_traj_name)
            os.rename(self.last_traj_path, new_traj_path)
            self.last_traj_path = new_traj_path
            self.traj_saved = False
    
    def set_action_space(self, action_space):
        self.env.set_action_space(action_space)

    def close(self):
        self.close_camera_feed()
        self.env.close()
        self.controller.close()
