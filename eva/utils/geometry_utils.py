import numpy as np
from scipy.spatial.transform import Rotation as R


### Conversions ###
def quat_to_euler(quat, degrees=False):
    euler = R.from_quat(quat).as_euler("xyz", degrees=degrees)
    return euler

def euler_to_quat(euler, degrees=False):
    return R.from_euler("xyz", euler, degrees=degrees).as_quat()

def rmat_to_euler(rot_mat, degrees=False):
    euler = R.from_matrix(rot_mat).as_euler("xyz", degrees=degrees)
    return euler

def euler_to_rmat(euler, degrees=False):
    return R.from_euler("xyz", euler, degrees=degrees).as_matrix()

def rmat_to_quat(rot_mat, degrees=False):
    quat = R.from_matrix(rot_mat).as_quat()
    return quat

def quat_to_rmat(quat, degrees=False):
    return R.from_quat(quat, degrees=degrees).as_matrix()

def rotvec_to_euler(rot_vec, degrees=False):
    return R.from_rotvec(rot_vec).as_euler("xyz", degrees=degrees)

def euler_to_rotvec(euler, degrees=False):
    return R.from_euler("xyz", euler, degrees=degrees).as_rotvec()

def rot6d_to_rmat(rot_6d):
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2)  # (..., 3, 3)

def rmat_to_rot6d(rot_mat):
    return rot_mat[..., :2, :].reshape(*rot_mat.shape[:-2], 6)

def rot6d_to_euler(rot_6d, convention="xyz", degrees=False):
    rot_mat = rot6d_to_rmat(rot_6d)
    return R.from_matrix(rot_mat).as_euler(convention, degrees=degrees)

def euler_to_rot6d(euler_angles, convention="xyz", degrees=False):
    rot_mat = R.from_euler(convention, euler_angles, degrees=degrees).as_matrix()
    return rmat_to_rot6d(rot_mat)


### Operations ###
def quat_diff(target, source):
    result = R.from_quat(target) * R.from_quat(source).inv()
    return result.as_quat()

def angle_diff(target, source, degrees=False):
    target_rot = R.from_euler("xyz", target, degrees=degrees)
    source_rot = R.from_euler("xyz", source, degrees=degrees)
    result = target_rot * source_rot.inv()
    return result.as_euler("xyz")

def pose_diff(target, source, degrees=False):
    target, source = np.array(target), np.array(source)
    lin_diff = target[..., :3] - source[..., :3]
    rot_diff = angle_diff(target[..., 3:6], source[..., 3:6], degrees=degrees)
    result = np.concatenate([lin_diff, rot_diff], axis=-1)
    return result

def add_quats(delta, source):
    result = R.from_quat(delta) * R.from_quat(source)
    return result.as_quat()

def add_angles(delta, source, degrees=False):
    delta_rot = R.from_euler("xyz", delta, degrees=degrees)
    source_rot = R.from_euler("xyz", source, degrees=degrees)
    new_rot = delta_rot * source_rot
    return new_rot.as_euler("xyz", degrees=degrees)

def add_poses(delta, source, degrees=False):
    delta, source = np.array(delta), np.array(source)
    lin_sum = delta[..., :3] + source[..., :3]
    rot_sum = add_angles(delta[..., 3:6], source[..., 3:6], degrees=degrees)
    result = np.concatenate([lin_sum, rot_sum], axis=-1)
    return result


### Transforms ###
def change_pose_frame(pose, frame, degrees=False):
    R_frame = euler_to_rmat(frame[3:6], degrees=degrees)
    R_pose = euler_to_rmat(pose[3:6], degrees=degrees)
    t_frame, t_pose = frame[:3], pose[:3]
    euler_new = rmat_to_euler(R_frame @ R_pose, degrees=degrees)
    t_new = R_frame @ t_pose + t_frame
    result = np.concatenate([t_new, euler_new])
    return result

def compose_transformation_matrix(pos, rot):
    pos, rot = np.array(pos), np.array(rot)
    T = np.eye(4) if len(pos.shape) == 1 else np.tile(np.eye(4), (pos.shape[0], 1, 1))
    T[..., :3, :3] = euler_to_rmat(rot)
    T[..., :3, 3] = pos
    return T

def decompose_transformation_matrix(T):
    pos = T[..., :3, 3]
    rot = rmat_to_euler(T[..., :3, :3])
    return pos, rot

def transform_world_to_camera(pos, rot, extrinsics):
    T_world = compose_transformation_matrix(pos, rot)
    T_camera = extrinsics @ T_world
    return decompose_transformation_matrix(T_camera)

def transform_trajectory_world_to_camera(trajectory, camera_extrinsics):
    camera_frame_trajectory = []
    for i in trajectory:
        pos_world, rot_world, gripper_state = i[:3], i[3:6], i[6:7]
        pos_cam, rot_cam = transform_world_to_camera(pos_world, rot_world, camera_extrinsics)
        camera_frame_trajectory.append(np.concatenate([pos_cam, rot_cam, gripper_state]))
    return np.array(camera_frame_trajectory)

def project_camera_to_image(pos, intrinsics):
    pos = pos.reshape(-1, 3)
    pixel = (intrinsics @ pos.T).T
    pixel = pixel / pixel[:, [2]]
    return pixel[:, :2] if len(pixel) > 1 else pixel[0, :2]

def project_world_coord_to_image(pos, intrinsics, extrinsics):
    rot = np.zeros(3) 
    pos_cam, _= transform_world_to_camera(pos, rot, extrinsics)
    pixel = project_camera_to_image(pos_cam, intrinsics)
    return pixel