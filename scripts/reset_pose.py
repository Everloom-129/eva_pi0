
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import threading

from franka_wliang.controllers.occulus import Occulus
from franka_wliang.env import FrankaEnv
from franka_wliang.runner import Runner


if __name__ == "__main__":
    env = FrankaEnv()
    controller = Occulus()
    runner = Runner(env=env, controller=controller)