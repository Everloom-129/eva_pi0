
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path
import cv2
import zerorpc
import base64
import io

from eva.controllers.controller import Controller
import eva.utils.parameters as params


class Proxy(Controller):
    def __init__(self, action_space="cartesian_velocity", gripper_action_space="velocity"):
        self.action_space = action_space
        self.gripper_action_space = gripper_action_space
        self.remote = zerorpc.Client(heartbeat=None, timeout=None)
        self.remote.connect("tcp://172.16.0.42:8787")
        print(f"Connected to remote controller at tcp://172.16.0.42:8787")

        self._state = {
            "success": False,
            "failure": False,
            "movement_enabled": True,
            "controller_on": True,
        }
    
    def forward(self, obs):
        robot_state = obs["robot_state"]
        obs = {
            "cart_pos": np.concatenate((robot_state["cartesian_position"], [robot_state["gripper_position"]])),
            "varied_camera_1": Image.fromarray(cv2.cvtColor(obs["image"][f"{params.varied_camera_1_id}_left"], cv2.COLOR_BGR2RGB)).convert("RGB"),
            "varied_camera_2": Image.fromarray(cv2.cvtColor(obs["image"][f"{params.varied_camera_2_id}_left"], cv2.COLOR_BGR2RGB)).convert("RGB"),
            "hand_camera": Image.fromarray(cv2.cvtColor(obs["image"][f"{params.hand_camera_id}_left"], cv2.COLOR_BGR2RGB)).convert("RGB"),
        }
        obs = {k: self.serialize(v) for k, v in obs.items()}
        action = self.remote.forward(obs)
        action = self.deserialize(action)
        return action, {}

    def serialize(self, x):
        if isinstance(x, np.ndarray):
            return ({
                "data": base64.b64encode(x.tobytes()).decode('utf-8'),
                "dtype": str(x.dtype),
                "shape": x.shape,
            }, 'numpy.ndarray')
        elif isinstance(x, Image.Image):
            buffer = io.BytesIO()
            x.save(buffer, format='PNG')
            return (base64.b64encode(buffer.getvalue()).decode('utf-8'), 'Image.Image')
        else:
            raise TypeError(f"Unsupported type for serialization: {type(x)}")
    
    def deserialize(self, x):
        x, data_type = x
        if data_type == 'numpy.ndarray':
            data = base64.b64decode(x['data'])
            return np.frombuffer(data, dtype=np.dtype(x['dtype'])).reshape(x['shape'])
        elif data_type == 'Image.Image':
            return Image.open(io.BytesIO(base64.b64decode(x)))
        else:
            raise TypeError(f"Unsupported type for deserialization: {data_type}")
        

    def reset_state(self):
        self._state = {
            "success": False,
            "failure": False,
            "movement_enabled": True,
            "controller_on": True,
        }
    
    def get_info(self):
        return self._state
    
    def register_key(self, key):
        if key == ord(" "):
            self._state["movement_enabled"] = not self._state["movement_enabled"]
            print("Movement enabled:", self._state["movement_enabled"])
        elif key == 13:
            self._state["success"] = True
        elif key == 8:
            self._state["failure"] = True
    
    def close(self):
        pass
    