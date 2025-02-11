
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
from datetime import datetime

from franka_wliang.controllers.occulus import Occulus
from franka_wliang.env import FrankaEnv
from franka_wliang.runner import Runner
from franka_wliang.utils.misc_utils import run_threaded_command, keyboard_listener, data_dir
from franka_wliang.manager import load_runner


def take_pictures(runner: Runner):
    try:
        camera_feed, cam_ids = runner.get_camera_feed()
    except:
        print("ERROR: Camera feed not available!")
    
    output_dir = data_dir / "images" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    for cam_id, feed in zip(cam_ids, camera_feed):
        im = cv2.cvtColor(feed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"{cam_id}.jpg"), im)
    print(f"Saved pictures to {output_dir}")


if __name__ == "__main__":
    runner = load_runner()
    take_pictures(runner)
