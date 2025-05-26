import time
import numpy as np
import functools
import zerorpc


def robust_call(relaunch=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            for i in range(3):
                try:
                    return func(self, *args, **kwargs)
                except zerorpc.exceptions.RemoteError as e:
                    print(f"[Attempt {i+1}] RemoteError: {e}. Retrying...")
                except zerorpc.exceptions.TimeoutExpired as e:
                    print(f"[Attempt {i+1}] TimeoutExpired: {e}. Retrying...")
                except Exception as e:
                    print(f"[Attempt {i+1}] Unexpected error: {e}. Retrying...")

                if relaunch:
                    try:
                        self.kill_controller()
                    except zerorpc.exceptions.RemoteError as e:
                        print(f"[Attempt {i+1}] RemoteError: {e}. Skipping kill_controller...")
                    except zerorpc.exceptions.TimeoutExpired as e:
                        print(f"[Attempt {i+1}] TimeoutExpired: {e}. Skipping kill_controller...")
                    except Exception as e:
                        print(f"[Attempt {i+1}] Unexpected error: {e}. Skipping kill_controller...")
                    self.launch_controller()
                    self.launch_robot()
                self.establish_connection(force=True)
                time.sleep(1)
            
            print("[Final Attempt] Trying one more time after 3 retries.")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


class ServerInterface:
    def __init__(self, ip_address="127.0.0.1", launch=True):
        self.ip_address = ip_address
        self.server = None
        self.establish_connection()
        if launch:
            self.launch_controller()
            self.launch_robot()

    def establish_connection(self, force=False):
        if not force:
            if self.server is not None:
                if self.server._zeromq_socket is not None:
                    print("Connection already established, skipping...")
                    return
                else:
                    print("Connection lost, re-establishing...")
                    self.server.close()
        else:
            if self.server is not None:
                self.server.close()
        self.server = zerorpc.Client(heartbeat=20, timeout=60)
        self.server.connect("tcp://" + self.ip_address + ":4242")
        print(f"Established connection to robot at {self.ip_address}:4242")

    @robust_call(relaunch=False)
    def launch_controller(self):
        self.server.launch_controller()

    @robust_call(relaunch=False)
    def launch_robot(self):
        self.server.launch_robot()

    @robust_call(relaunch=False)
    def kill_controller(self):
        self.server.kill_controller()

    @robust_call()
    def update_command(self, command, action_space="cartesian_velocity", gripper_action_space="velocity", blocking=False):
        action_dict = self.server.update_command(command.tolist(), action_space, gripper_action_space, blocking)
        return action_dict

    @robust_call()
    def create_action_dict(self, command, action_space="cartesian_velocity"):
        action_dict = self.server.create_action_dict(command.tolist(), action_space)
        return action_dict

    @robust_call()
    def update_pose(self, command, velocity=True, blocking=False):
        self.server.update_pose(command.tolist(), velocity, blocking)

    @robust_call()
    def update_joints(self, command, velocity=True, blocking=False, cartesian_noise=None):
        if cartesian_noise is not None:
            cartesian_noise = cartesian_noise.tolist()
        self.server.update_joints(command.tolist(), velocity, blocking, cartesian_noise)

    @robust_call()
    def update_gripper(self, command, velocity=True, blocking=False):
        self.server.update_gripper(command, velocity, blocking)

    @robust_call()
    def get_ee_pose(self):
        return np.array(self.server.get_ee_pose())

    @robust_call()
    def get_joint_positions(self):
        return np.array(self.server.get_joint_positions())

    @robust_call()
    def get_joint_velocities(self):
        return np.array(self.server.get_joint_velocities())

    @robust_call()
    def get_gripper_state(self):
        return self.server.get_gripper_state()

    @robust_call()
    def get_robot_state(self):
        return self.server.get_robot_state()
    
    def close(self):
        self.server.close()
