import numpy as np
import cv2
import os
import shutil
from datetime import datetime
import json
import re
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import subprocess
import scipy

from franka_wliang.utils.calibration_utils import load_calibration_info
from franka_wliang.utils.geometry_utils import euler_to_rmat, rmat_to_euler, compose_transformation_matrix, decompose_transformation_matrix

def draw_axis(pos, rot, intrinsics, scale=0.05):
    axes = scale * np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    colors_bgr = [
        (0, 0, 255, 255),
        (0, 255, 0, 255),
        (255, 0, 0, 255)
    ]

    origin_pixel = intrinsics @ pos
    origin_pixel = origin_pixel / origin_pixel[2]
    origin_pixel = tuple(map(int, origin_pixel[:2]))
    
    axis_overlay = np.zeros((720, 1280, 4), dtype=np.uint8)
    for axis, color in zip(axes, colors_bgr):
        endpoint = pos + euler_to_rmat(rot) @ axis * 2
        endpoint_pixel = intrinsics @ endpoint
        endpoint_pixel = endpoint_pixel / endpoint_pixel[2]
        endpoint_pixel = tuple(map(int, endpoint_pixel[:2]))
        cv2.arrowedLine(axis_overlay, origin_pixel, endpoint_pixel, color, 2, tipLength=0.2)
    return axis_overlay


def create_spline(trajectory):
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    tck, u = scipy.interpolate.splprep([x, y, z], s=0)
    return tck


def douglas_peucker(points, epsilon=0.25):
    """Downsamples a curve into keypoints, ensuring that maximum distance error is less than epsilon."""
    def recurse(points, indices, epsilon):
        # Find the point with the maximum distance
        start, end = points[0], points[-1]
        line_vec = end - start
        line_norm = np.linalg.norm(line_vec)
        line_unit_vec = line_vec / line_norm
        distances = np.abs(np.cross(points - start, line_unit_vec)) / line_norm

        max_dist_idx = np.argmax(distances)
        max_dist = distances[max_dist_idx]

        if max_dist > epsilon:
            # Recursive call for each segment and track the indices
            left_points, left_indices = recurse(points[:max_dist_idx + 1], indices[:max_dist_idx + 1], epsilon)
            right_points, right_indices = recurse(points[max_dist_idx:], indices[max_dist_idx:], epsilon)
            return np.vstack((left_points[:-1], right_points)), np.concatenate((left_indices[:-1], right_indices))
        else:
            return np.vstack((start, end)), np.array([indices[0], indices[-1]])

    # Initialize indices
    indices = np.arange(len(points))
    selected_points, selected_indices = recurse(points, indices, epsilon)
    return selected_points, selected_indices


def draw_circle_text(img, x, y, text, color=(255, 255, 255, 255)):
    cv2.circle(img, (x, y), 16, color, -1)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x, text_y = x - text_size[0] // 2, y + text_size[1] // 2
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0, 255), 2)
    return img


def plot_trajectory(input_dir, densify_factor=64):
    output_dir = input_dir / "plots"
    frames_dir = output_dir / "frames"
    keyframes_dir = output_dir / "keyframes"
    output_dir.mkdir(exist_ok=True)
    frames_dir.mkdir(exist_ok=True)
    keyframes_dir.mkdir(exist_ok=True)

    camera = "27085680"  # TODO remove hardcoding
    calibration_dict = load_calibration_info()
    camera_extrinsics_vec = calibration_dict[f"{camera}_left"]
    camera_extrinsics = np.eye(4)
    camera_extrinsics[:3, :3] = euler_to_rmat(camera_extrinsics_vec[3:])
    camera_extrinsics[:3, 3] = camera_extrinsics_vec[:3]
    
    camera_intrinsics = np.array([[528.65686035,   0.        , 636.79156494],
                                 [  0.        , 528.65686035, 372.25308228],
                                 [  0.        ,   0.        ,   1.        ]])

    traj_info = np.load(input_dir / "trajectory.npy", allow_pickle=True).item()

    # Transform trajectory from world to camera frame
    camera_frame_trajectory = []
    for i in traj_info["actions_pos"]:
        pos_world, rot_world, gripper_state = i[:3], i[3:6], i[6:7]
        T_world = compose_transformation_matrix(pos_world, rot_world)
        T_camera = np.linalg.inv(camera_extrinsics) @ T_world
        pos_cam, rot_cam = decompose_transformation_matrix(T_camera)
        camera_frame_trajectory.append(np.concatenate([pos_cam, rot_cam, gripper_state]))
    camera_frame_trajectory = np.array(camera_frame_trajectory)

    # Project trajectory to image
    image_trajectory = []
    for i in camera_frame_trajectory:
        pos = i[:3]
        pixel_coords = camera_intrinsics @ pos
        pixel_coords = pixel_coords / pixel_coords[2]
        image_trajectory.append(pixel_coords[:2])
    image_trajectory = np.array(image_trajectory)

    # Compute splined trajectory in image space
    # TODO avoid repeat code with above
    camera_frame_spline_trajectory = create_spline(camera_frame_trajectory)
    image_spline_trajectory = []
    for i in np.linspace(0, 1, len(camera_frame_trajectory) * densify_factor):
        # TODO this should be interpolating temporally, not spatially
        pos = np.array(scipy.interpolate.splev(i, camera_frame_spline_trajectory))
        pixel_coords = camera_intrinsics @ pos
        pixel_coords = pixel_coords / pixel_coords[2]
        image_spline_trajectory.append(pixel_coords[:2])
    image_spline_trajectory = np.array(image_spline_trajectory)

    # Create (splined) trajectory overlay
    trajectory_overlay = np.zeros((720, 1280, 4), dtype=np.uint8)
    cmap = plt.cm.plasma(np.linspace(0, 1, len(image_spline_trajectory)))
    for i, pixel in enumerate(image_spline_trajectory):
        x, y = int(pixel[0]), int(pixel[1])
        if 0 <= x < 1280 and 0 <= y < 720:
            color = tuple(map(int, cmap[i] * 255))
            cv2.circle(trajectory_overlay, (x, y), 2, color, -1)
    trajectory_overlay = Image.fromarray(trajectory_overlay)
    
    # Create keypoint overlay
    keypoint_overlay = np.zeros((720, 1280, 4), dtype=np.uint8)
    downsampled_image_trajectory, downsampled_indices = douglas_peucker(image_trajectory)
    cmap = plt.cm.plasma(np.linspace(0, 1, len(image_trajectory)))
    for i, pixel in enumerate(downsampled_image_trajectory):
        x, y = int(pixel[0]), int(pixel[1])
        if 0 <= x < 1280 and 0 <= y < 720:
            color = tuple(map(int, cmap[downsampled_indices[i]] * 255))
            draw_circle_text(keypoint_overlay, x, y, f"{i:02d}", color=color)
    keypoint_overlay = Image.fromarray(keypoint_overlay)

    # Create axis overlay and save annotated frames
    for i in range(len(camera_frame_trajectory)):
        frame = Image.open(input_dir / "recordings" / "frames" / "varied_camera_2" / f"{i:03d}.jpg")
        axis_overlay = draw_axis(camera_frame_trajectory[i][:3], camera_frame_trajectory[i][3:6], camera_intrinsics)
        axis_overlay = Image.fromarray(axis_overlay)
        frame.paste(trajectory_overlay, (0, 0), trajectory_overlay)
        frame.paste(axis_overlay, (0, 0), axis_overlay)
        frame.paste(keypoint_overlay, (0, 0), keypoint_overlay)
        frame.save(str(frames_dir / f"{i:03d}.jpg"))
        if i in downsampled_indices:
            frame.save(str(keyframes_dir / f"{i:03d}.jpg"))

    keypoint_trajectory = trajectory_overlay.copy()
    keypoint_trajectory.paste(keypoint_overlay, (0, 0), keypoint_overlay)
    keypoint_trajectory.save(str(output_dir / "trajectory.png"))

    subprocess.run([
        "ffmpeg", "-y", "-framerate", "15", "-i", str(frames_dir / f"%03d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_dir / f"video.mp4")
    ])
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    args = parser.parse_args()

    plot_trajectory(Path(args.input_dir))
