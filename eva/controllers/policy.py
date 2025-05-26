
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path
import cv2

from eva.controllers.controller import Controller


class Policy(Controller):
    def __init__(self, policy_path=None, action_space="cartesian_velocity", gripper_action_space="velocity"):
        self.cfg = OmegaConf.load(Path(policy_path).parent.parent / ".hydra" / "config.yaml")
        print(OmegaConf.to_yaml(self.cfg))
        self.action_space = action_space
        self.gripper_action_space = gripper_action_space
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.policy = torch.nn.Sequential(
            # Fill this in with your policy architecture
        )
        checkpoint = torch.load(policy_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint)
        self.policy.eval()

        self.image_transform = T.Compose([
            T.RandomVerticalFlip(p=1.0),
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
        ])

        self._state = {
            "success": False,
            "failure": False,
            "movement_enabled": True,
            "controller_on": True,
        }
    
    def forward(self, obs):
        robot_state = obs["robot_state"]
        robot_state = np.concatenate((robot_state["cartesian_position"], [robot_state["gripper_position"]]))
        img = Image.fromarray(cv2.cvtColor(obs["image"]["15512737_left"], cv2.COLOR_BGR2RGB)).convert("RGB")

        obs = {
            "robot_state": robot_state,
            "hand_image": self.image_transform(img),
        }
        obs = {k: torch.tensor(v).unsqueeze(0).to(self.device) for k, v in obs.items()}

        with torch.no_grad():
            # Replace this with your policy inference
            pred = self.policy(obs["robot_state"])
            pred = pred.squeeze(0).cpu().numpy()

        return pred, {}

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
    
    def close(self):
        pass
    