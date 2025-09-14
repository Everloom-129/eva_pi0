import numpy as np
import time
import threading
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from eva.utils.misc_utils import print_datadict_shape, print_datadict_tree, create_info_dict
from dataclasses import dataclass
from eva.controllers.pi0_policy import Pi0Policy, Pi0PolicyConfig
from eva.controllers.spacemouse import SpaceMouse
import eva.utils.parameters as params
from colorama import Fore, init
init(autoreset=True)

'''
SpaceMouse-Pi0 Mixed controller for EVA framework
Author: Jie Wang
Version: 
   2025-04-12: initiate from pi0 policy controller
   2025-04-13: add proper action space handling and controller switching
   2025-04-16: debug the action recording of cartesian_velocity and joint_velocity
'''
@dataclass
class SpacemousePi0Config:
    # PI0 config
    init_instruction: str = "find the pineapple and pick it up"
    remote_host: str = "158.130.52.14"
    remote_port: int = 8000
    action_space: str = "joint_velocity"
    gripper_action_space: str = "position"
    left_camera_id: str = params.varied_camera_1_id
    right_camera_id: str = params.varied_camera_2_id
    wrist_camera_id: str = params.hand_camera_id
    external_camera: str = "left"
    open_loop_horizon: int = 8
    
    # SpaceMouse config
    max_lin_vel: float = 3
    max_rot_vel: float = 3
    max_gripper_vel: float = 3
    pos_sensitivity: float = 8.0
    rot_sensitivity: float = 8.0
    action_scale: float = 0.15
    deadzone: float = 0.05
    smoothing: float = 0.3

class SpacemousePi0:
    def __init__(self, config: SpacemousePi0Config = SpacemousePi0Config(), on_switch_callback=None):
        """Initialize both controllers and set up switching logic"""
        print(Fore.BLUE + "Initializing SpacemousePi0 Controller ")
        print(Fore.BLUE + "SpacemousePi0 == CONFIG == \n notice you need to change it under eva/controllers/pi0_spacemouse.py:\n")
        
        # Callback for notifying action space changes
        self._on_switch_callback = on_switch_callback
        self.init_instruction = config.init_instruction
        
        # Add list to store cartesian velocity actions
        self._cartesian_velocity_actions = []
        
        # Initialize PI0 controller (but don't start querying yet)
        pi0_config = Pi0PolicyConfig(
            remote_host=config.remote_host,
            remote_port=config.remote_port,
            action_space=config.action_space,
            gripper_action_space=config.gripper_action_space,
            left_camera_id=config.left_camera_id,
            right_camera_id=config.right_camera_id,
            wrist_camera_id=config.wrist_camera_id,
            external_camera=config.external_camera,
            open_loop_horizon=config.open_loop_horizon
        )
        self.pi0_controller = Pi0Policy(pi0_config)
        
        # Initialize SpaceMouse controller
        # TODO : error caused by the runner initialization will also activate the SpaceMouse listener thread
        #  - find thread to connect to it?
        #  - how will do this?
        self.spacemouse_controller = SpaceMouse(
            max_lin_vel=config.max_lin_vel,
            max_rot_vel=config.max_rot_vel,
            max_gripper_vel=config.max_gripper_vel,
            pos_sensitivity=config.pos_sensitivity,
            rot_sensitivity=config.rot_sensitivity,
            action_scale=config.action_scale,
            deadzone=config.deadzone,
            smoothing=config.smoothing
        )
        
        # Controller switching lock
        self._switch_lock = threading.Lock()
        self._switching_in_progress = False
        
        self.reset_state() # initialize spacemousepi0 controller's internal state
        print("\n\n ==== SpacemousePi0 Controller Controls: ====")
        print("- '=' key: Switch between PI0 and SpaceMouse")
        print("Current controller: PI0")
        
        # Start with PI0
        # self.pi0_controller.start_policy()
        print("SpacemousePi0 == INIT FINISHED ==")
        
    @property
    def action_space(self):
        """Dynamically return current controller's action space"""
        return self.get_current_controller().action_space
    
    @property
    def gripper_action_space(self):
        """Dynamically return current controller's gripper action space"""
        return self.get_current_controller().gripper_action_space
        
    def get_name(self):
        return "spacemouse_pi0"
        
    def reset_state(self):
        """Reset both controllers and internal state"""
        self.pi0_controller.reset_state()
        self.spacemouse_controller.reset_state()
        self.pi0_controller.stop_policy()
        
        # Clear recorded actions
        self._cartesian_velocity_actions = []

        self._state = {
            "success": False,
            "failure": False,
            "movement_enabled": True,
            "controller_on": True,
            "switch_label": 1,  # 0: PI0, 1: SpaceMouse, 0.5: Transitioning
            "current_controller": 1   #
        }
        
        if self._on_switch_callback:
            current_controller = self.spacemouse_controller
            print(f"Current controller: {current_controller.get_name()}")
            print(f"Action space: {current_controller.action_space}")
            print(f"Gripper action space: {current_controller.gripper_action_space}")
            self._on_switch_callback(
                current_controller.action_space,
                current_controller.gripper_action_space
            )
        
        self.set_instruction(self.init_instruction)

    def set_instruction(self, instruction):
        """Set instruction for PI0 controller"""
        self.pi0_controller.set_instruction(instruction)
        
    def register_key(self, key):
        """Handle key presses for both controllers"""
        if key == ord("="):
            with self._switch_lock:
                if not self._switching_in_progress:
                    self._switching_in_progress = True
                    # Switch controller
                    if self._state["current_controller"] == 0:
                        print("Stopping PI0 policy...")
                        
                        print("Preparing to switch to SpaceMouse controller...")
                        time.sleep(0.1)  # Small delay to ensure clean switch
                        
                        self._state["switch_label"] = 1.0
                        # input("Press Enter to continue...")
                        self.pi0_controller.stop_policy()
                        self._state["current_controller"] = 1

                        print("Switched to SpaceMouse controller")
                    else:
                        self.save_cartesian_actions("data/spacemouse_actions.npz")

                        self._state["switch_label"] = 0.0
                        print("Preparing to switch to PI0 controller...")
                        time.sleep(0.1)  # Small delay to ensure clean switch
                        
                        print("Starting PI0 policy...")
                        # input("Press Enter to continue...")
                        self.pi0_controller.start_policy()
                        
                        self._state["current_controller"] = 0
                        print("Switched to PI0 controller")
                        # self.pi0_controller.set_instruction(input("Enter command for PI0: "))

                    # Notify runner about action space change
                    if self._on_switch_callback:
                        current_controller = self.get_current_controller()
                        print(f"Current controller: {current_controller.get_name()}")
                        print(f"Action space: {current_controller.action_space}")
                        print(f"Gripper action space: {current_controller.gripper_action_space}")
                        self._on_switch_callback(
                            current_controller.action_space,
                            current_controller.gripper_action_space
                        )
                    
                    self._switching_in_progress = False
        
        # Forward key press to current controller
        if self._state["current_controller"] == 0:
            self.pi0_controller.register_key(key)
        else:
            self.spacemouse_controller.register_key(key)
            
        # Update shared state
        controller_state = self.get_current_controller().get_info()
        self._state["success"] = controller_state["success"]
        self._state["failure"] = controller_state["failure"]
        self._state["movement_enabled"] = controller_state["movement_enabled"]
        self._state["controller_on"] = controller_state["controller_on"]
    
    def get_info(self):
        """Get combined controller info"""
        return self._state
    
    def get_current_controller(self):
        """Helper to get current active controller"""
        return self.pi0_controller if self._state["current_controller"] == 0 else self.spacemouse_controller
    
    def forward(self, observation):
        """Forward observation to current controller"""
        # print_datadict_shape(observation, indent=4, save_data=True)

        with self._switch_lock:
            if self._state["current_controller"] == 0: # PI0 controller， 8 joint + 1 gripper
                # print('pi00000000')
                
                action, info = self.pi0_controller.forward(observation)
                info["joint_velocity"] = action[:7]
                info["gripper_velocity"] = action[-1]

            else: # SpaceMouse controller, 3D translation + 3D rotation + 1 gripper
                # print('spacemouse00000000')
                action, info = self.spacemouse_controller.forward(observation)
                info["cartesian_velocity"] = action[:6]
                info["gripper_velocity"] = action[-1]
                
                # Record cartesian velocity action if movement is enabled
                if self._state["movement_enabled"]:
                    self._cartesian_velocity_actions.append(action)

            if info is None:
                print("MIX == INFO IS NONE == ")
            
            # Ensure action is float32
            action = np.array(action, dtype=np.float32)

            # print_datadict_shape(info, indent=4, save_data=True)
            action = np.clip(action, -1, 1)
            return action, info
    
    def close(self):
        """Clean up both controllers"""
        self.pi0_controller.close()
        self.spacemouse_controller.close()

    def save_cartesian_actions(self, filepath):
        """Save recorded cartesian velocity actions to a .npz file"""
        if self._cartesian_velocity_actions:
            data_dict = {
                "states": np.zeros_like(self._cartesian_velocity_actions),
                "actions_pos": np.zeros_like(self._cartesian_velocity_actions),
                "actions_vel": np.array(self._cartesian_velocity_actions)
            }
            np.savez(filepath, **data_dict)
            print(f"Saved {len(self._cartesian_velocity_actions)} cartesian velocity actions to {filepath}")
        else:
            print("No cartesian velocity actions recorded")