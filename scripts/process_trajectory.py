import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import csv
import os
import argparse
from pathlib import Path
import subprocess
from collections import defaultdict

from franka_wliang.utils.trajectory_utils import load_trajectory
from franka_wliang.utils.misc_utils import get_latest_trajectory
from franka_wliang.data_processing.timestep_processor import TimestepProcessor
from PIL import Image
from collections import defaultdict

def extract_image(trajectory, camera_type):
    im = trajectory["observation"]["camera"]["image"][camera_type][0]
    im = Image.fromarray(im[:, :, :3])
    return im

def extract_depth(trajectory, camera_type):
    raw_depth = trajectory["observation"]["camera"]["depth"][camera_type][0]
    depth_array = np.nan_to_num(raw_depth, nan=0)
    return depth_array

def extract_pcd(trajectory, camera_type):
    pcd = trajectory["observation"]["camera"]["pointcloud"][camera_type][0]
    return pcd

def process_data(input_path, process_depth=False, process_pcd=False):
    if input_path is None:
        data_dirs = [get_latest_trajectory()]
        print(f"Processing latest trajectory: {data_dirs[0]}")
    else:
        data_dirs = []
        if os.path.exists(os.path.join(input_path, "trajectory.h5")):
            data_dirs.append(input_path)
    # else:
    #     data_dirs = glob.glob(str(input_path) + "*/**/", recursive=True)
    #     data_dirs = [d for d in data_dirs if os.path.exists(os.path.join(d, "trajectory.h5"))]

    image_transform = {"remove_alpha": True, "bgr_to_rgb": True, "augment": False}
    camera_kwargs = defaultdict(
        lambda: {"depth": process_depth, "pointcloud": process_pcd}
    )

    timestep_processer = TimestepProcessor(
        camera_extrinsics=["fixed_camera", "hand_camera", "varied_camera"],
        image_transform_kwargs=image_transform,
    )
    

    for traj_dir in data_dirs:
        traj_dir = Path(traj_dir)
        tqdm.write(str(traj_dir))
        print(traj_dir)

        filepath = traj_dir / "trajectory.h5"
        recording_folderpath = traj_dir / "recordings"
        if not recording_folderpath.exists():
            recording_folderpath = None

        try:
            traj_samples = load_trajectory(
                str(filepath),
                recording_folderpath=str(recording_folderpath),
                camera_kwargs=camera_kwargs,
                remove_skipped_steps=True,
            )
        except Exception as e:
            raise e
            tqdm.write(f"WARNING: Failed to load trajectory at {traj_dir}")
            continue
        traj = [timestep_processer.forward(t) for t in traj_samples]
        if len(traj) == 0:
            tqdm.write(f"WARNING: Empty trajectory at {traj_dir}")
            continue

        states = []
        actions_pos = []
        actions_vel = []
        depths = []
        pointclouds = []

        frames_dir = traj_dir / "recordings" / "frames"
        frames_dir.mkdir(exist_ok=True)
        if process_depth:
            depth_dir = traj_dir / "depths"
            depth_dir.mkdir(exist_ok=True)

        if process_pcd:
            pcd_dir = traj_dir / "point_clouds"
            pcd_dir.mkdir(exist_ok=True)

        videos_dir = traj_dir / "recordings" / "videos"
        videos_dir.mkdir(exist_ok=True)

        camera_types = traj[0]["observation"]["camera"]["image"].keys()
        for camera_type in camera_types:
            (frames_dir / camera_type).mkdir(exist_ok=True)
            if process_depth:
                (depth_dir / camera_type).mkdir(exist_ok=True)
            if process_pcd:
                (pcd_dir / camera_type).mkdir(exist_ok=True)
        for t in tqdm(range(len(traj))):
            step_t = traj[t]
            states.append(step_t["observation"]["state"])
            actions_pos.append(step_t["action"]["cartesian_position"])
            try:
                actions_vel.append(step_t["action"]["cartesian_velocity"])
            except KeyError:
                print("No action velocity saved. You are probably using cartesian_position mode.")
            # depths.append(step_t['observation']['camera']['depth']['fixed_camera'][0])
            # pointclouds.append(step_t['observation']['camera']['pointcloud']['fixed_camera'][0])
            
            
            #rgbd_array_normalized = np.nan_to_num(step_t['observation']['camera']['pointcloud']['fixed_camera'][0], nan=0)
            #rgbd_array_normalized = (rgbd_array_normalized / np.nanmax(rgbd_array_normalized)) * 255

            if "depth" in traj[t]["observation"]["camera"]:
                depth_array = np.nan_to_num(step_t['observation']['camera']['depth'], nan=0)
                dpth_array_min = np.min(depth_array)
                dpth_array_max = np.max(depth_array)
                depth_array_normalized = ((depth_array - dpth_array_min) / (dpth_array_max - dpth_array_min)) * 255
                depth_image = depth_array_normalized.astype(np.uint8)
                image = Image.fromarray(depth_image)
                image.save(f"{frames_dir}/{t}_depth_image.png")
            
            for camera_type in camera_types:
                im = extract_image(traj[t], camera_type)
                im.save(frames_dir / camera_type / f"{t:03d}.jpg")

                if process_depth:
                    depth = extract_depth(traj[t], camera_type)
                    np.save(depth_dir / camera_type / f"{t:03d}.npy", depth)
                
                if process_pcd:
                    pcd = extract_pcd(traj[t], camera_type)
                    np.save(pcd_dir / camera_type / f"{t:03d}.npy", pcd)

        trajectory = {"states": np.array(states), "actions_pos": np.array(actions_pos), "actions_vel": np.array(actions_vel), "text": ""} #, "depths": np.array(depths), "pointclouds": np.array(pointclouds) }
        np.savez(f"{traj_dir}/trajectory.npz", **trajectory)

        for camera_type in camera_types:
            subprocess.run([
                "ffmpeg", "-y", "-framerate", "15", "-i", str(frames_dir / camera_type / f"%03d.jpg"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(videos_dir / f"{camera_type}.mp4")
            ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, nargs="?", default=None)
    parser.add_argument("--process_depth", action="store_true")
    parser.add_argument("--process_pcd", action="store_true")
    args = parser.parse_args()

    path = None if args.input_path is None else Path(args.input_path)
    process_data(path, args.process_depth, args.process_pcd)
