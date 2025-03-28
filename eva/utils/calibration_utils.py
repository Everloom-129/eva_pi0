import json
import os
import time
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2 import aruco
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import shutil

from eva.utils.parameters import hand_camera_id, ARUCO_DICT, CHARUCOBOARD_ROWCOUNT, CHARUCOBOARD_COLCOUNT, CHARUCOBOARD_CHECKER_SIZE, CHARUCOBOARD_MARKER_SIZE
from eva.utils.geometry_utils import pose_diff, change_pose_frame, euler_to_rmat, project_camera_to_image, transform_world_to_camera, compose_transformation_matrix

# Create Board #
CHARUCO_BOARD = aruco.CharucoBoard_create(
    squaresX=CHARUCOBOARD_COLCOUNT,
    squaresY=CHARUCOBOARD_ROWCOUNT,
    squareLength=CHARUCOBOARD_CHECKER_SIZE,
    markerLength=CHARUCOBOARD_MARKER_SIZE,
    dictionary=ARUCO_DICT,
)

# Detector Params
detector_params = cv2.aruco.DetectorParameters_create()
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
calib_flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_FOCAL_LENGTH

# Prepare Calibration Info #
dir_path = os.path.dirname(os.path.realpath(__file__))
calib_info_filepath = os.path.join(dir_path, "calibration.json")


def load_calibration_info(keep_time=False):
    if not os.path.isfile(calib_info_filepath):
        return {}
    with open(calib_info_filepath, "r") as jsonFile:
        calibration_info = json.load(jsonFile)
    if not keep_time:
        calibration_info = {key: data["extrinsics"] for key, data in calibration_info.items()}
    return calibration_info


def update_calibration_info(cam_id, intrinsics, extrinsics):
    calibration_info = load_calibration_info(keep_time=True)
    calibration_info[cam_id] = {"intrinsics": list(intrinsics), "extrinsics": list(extrinsics), "timestamp": time.time()}
    with open(calib_info_filepath, "w") as jsonFile:
        json.dump(calibration_info, jsonFile)


def check_calibration_info(required_ids, time_threshold=3600):
    calibration_info = load_calibration_info(keep_time=True)
    calibration_ids = list(calibration_info.keys())
    info_dict = {"missing": [], "old": []}

    for cam_id in required_ids:
        if cam_id not in calibration_ids:
            info_dict["missing"].append(cam_id)
            continue
        time_passed = time.time() - calibration_info[cam_id]["timestamp"]
        if time_passed > time_threshold:
            info_dict["old"].append(cam_id)

    return info_dict


def save_calibration_info(save_path):
    shutil.copyfile(calib_info_filepath, save_path)


def visualize_calibration(calibration_dict):
    shapes = [".", "o", "v", "^", "s", "x", "D", "h", "<", ">", "8", "1", "2", "3"]
    assert len(calibration_dict) < (len(shapes) - 1)
    plt.clf()

    axes = plt.subplot(111, projection="3d")
    axes.plot(0, 0, 0, "*", label="Robot Base")

    for view_id in calibration_dict:
        curr_shape = shapes.pop(0)
        pose = calibration_dict[view_id]
        angle = [int(d * 180 / np.pi) for d in pose[3:]]
        label = "{0}: {1}".format(view_id, angle)
        axes.plot(pose[0], pose[1], pose[2], curr_shape, label=label)

    plt.legend(loc="center right", bbox_to_anchor=(2, 0.5))
    plt.title("Calibration Visualization")
    plt.show()


def calibration_traj(t, pos_scale=0.1, angle_scale=0.2, hand_camera=False):
    x = -np.abs(np.sin(3 * t)) * pos_scale
    y = -0.8 * np.sin(2 * t) * pos_scale
    z = 0.5 * np.sin(4 * t) * pos_scale
    a = -np.sin(4 * t) * angle_scale
    b = np.sin(3 * t) * angle_scale
    c = np.sin(2 * t) * angle_scale
    if hand_camera:
        value = np.array([z, y, -x, c / 1.5, b / 1.5, -a / 1.5])
    else:
        value = np.array([x, y, z, a, b, c])
    return value


class CharucoDetector:
    def __init__(
        self,
        intrinsics_dict,
        inlier_error_threshold=3.0,
        reprojection_error_threshold=3.0,
        num_img_threshold=10,
        num_corner_threshold=10,
    ):
        # Set Parameters
        self.inlier_error_threshold = inlier_error_threshold
        self.reprojection_error_threshold = reprojection_error_threshold
        self.num_img_threshold = num_img_threshold
        self.num_corner_threshold = num_corner_threshold
        self.intrinsic_params = {}
        self._intrinsics_dict = intrinsics_dict
        self._readings_dict = defaultdict(list)
        self._pose_dict = defaultdict(list)
        self._curr_cam_id = None

    def process_image(self, image):
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError
        img_size = image.shape[:2]

        # Find Aruco Markers In Image #
        corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=ARUCO_DICT, parameters=detector_params)

        corners, ids, _, _ = cv2.aruco.refineDetectedMarkers(
            gray,
            CHARUCO_BOARD,
            corners,
            ids,
            rejected,
            parameters=detector_params,
            **self._intrinsics_dict[self._curr_cam_id],
        )

        # Find Charuco Corners #
        if len(corners) == 0:
            return None

        num_corners_found, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners, markerIds=ids, image=gray, board=CHARUCO_BOARD, **self.intrinsic_params
        )

        if num_corners_found < self.num_corner_threshold:
            return None

        return corners, charuco_corners, charuco_ids, img_size

    def add_sample(self, cam_id, image, pose):
        readings = self.process_image(image)
        if readings is None:
            return
        self._readings_dict[cam_id].append(readings)
        self._pose_dict[cam_id].append(pose)

    def calculate_target_to_cam(self, readings, train=True):
        init_corners_all = []  # Corners discovered in all images processed
        init_ids_all = []  # Aruco ids corresponding to corners discovered
        fixed_image_size = readings[0][3]

        # Proccess Readings #
        init_successes = []
        for i in range(len(readings)):
            corners, charuco_corners, charuco_ids, img_size = readings[i]
            assert img_size == fixed_image_size
            init_corners_all.append(charuco_corners)
            init_ids_all.append(charuco_ids)
            init_successes.append(i)

        # First Pass: Find Outliers #
        threshold = self.num_img_threshold if train else 5
        if len(init_successes) < threshold:
            return None
        # print('Not enough points round 1')
        # print('Num Points: ', len(init_successes))
        # return None

        calibration_error, cameraMatrix, distCoeffs, rvecs, tvecs, stdIntrinsics, stdExtrinsics, perViewErrors = (
            aruco.calibrateCameraCharucoExtended(
                charucoCorners=init_corners_all,
                charucoIds=init_ids_all,
                board=CHARUCO_BOARD,
                imageSize=fixed_image_size,
                flags=calib_flags,
                **self._intrinsics_dict[self._curr_cam_id],
            )
        )

        # Remove Outliers #
        threshold = self.num_img_threshold if train else 5
        final_corners_all = [
            init_corners_all[i] for i in range(len(perViewErrors)) if perViewErrors[i] <= self.inlier_error_threshold
        ]
        final_ids_all = [
            init_ids_all[i] for i in range(len(perViewErrors)) if perViewErrors[i] <= self.inlier_error_threshold
        ]
        final_successes = [
            init_successes[i] for i in range(len(perViewErrors)) if perViewErrors[i] <= self.inlier_error_threshold
        ]
        if len(final_successes) < threshold:
            return None
        # print('Not enough points round 2')
        # print('Num Points: ', len(final_successes))
        # print('Error Mean: ', perViewErrors.mean())
        # print('Error Std: ', perViewErrors.std())
        # return None

        # Second Pass: Calculate Finalized Extrinsics #
        calibration_error, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
            charucoCorners=final_corners_all,
            charucoIds=final_ids_all,
            board=CHARUCO_BOARD,
            imageSize=fixed_image_size,
            flags=calib_flags,
            **self._intrinsics_dict[self._curr_cam_id],
        )

        # Return Transformation #
        if calibration_error > self.reprojection_error_threshold:
            return None
        # print('Failed Calibration Threshold')
        # print('Calibration Error: ', calibration_error)
        # return None

        rmats = [R.from_rotvec(rvec.flatten()).as_matrix() for rvec in rvecs]
        tvecs = [tvec.flatten() for tvec in tvecs]

        return rmats, tvecs, final_successes

    def augment_image(self, cam_id, image, visualize=False, visual_type=["markers", "axes"]):
        if type(visual_type) != list:
            visual_type = [visual_type]
        assert all([t in ["markers", "charuco", "axes"] for t in visual_type])
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        self._curr_cam_id = cam_id

        image = np.copy(image)
        readings = self.process_image(image)

        if readings is None:
            if visualize:
                cv2.imshow("Charuco board: {0}".format(cam_id), image)
                cv2.waitKey(20)
            return image

        corners, charuco_corners, charuco_ids, image_size = readings

        # Outline the aruco markers found in our query image
        if "markers" in visual_type:
            image = aruco.drawDetectedMarkers(image=image, corners=corners)

        # Draw the Charuco board we've detected to show our calibrator the board was properly detected
        if "charuco" in visual_type:
            image = aruco.drawDetectedCornersCharuco(image=image, charucoCorners=charuco_corners, charucoIds=charuco_ids)

        if "axes" in visual_type:
            calibration_error, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
                charucoCorners=[charuco_corners],
                charucoIds=[charuco_ids],
                board=CHARUCO_BOARD,
                imageSize=image_size,
                flags=calib_flags,
                **self._intrinsics_dict[self._curr_cam_id],
            )
            cv2.drawFrameAxes(image, cameraMatrix, distCoeffs, rvecs[0], tvecs[0], 0.1)

        # Visualize
        if visualize:
            cv2.imshow("Charuco board: {0}".format(cam_id), image)
            cv2.waitKey(20)

        return image


class ThirdPersonCameraCalibrator(CharucoDetector):
    def __init__(
        self, intrinsics_dict, lin_error_threshold=1e-3, rot_error_threshold=1e-2, train_percentage=0.7, **kwargs
    ):
        self.lin_error_threshold = lin_error_threshold
        self.rot_error_threshold = rot_error_threshold
        self.train_percentage = train_percentage
        super().__init__(intrinsics_dict, **kwargs)

    def calibrate(self, cam_id):
        return self._calibrate_cam_to_base(cam_id=cam_id)

    def _calibrate_cam_to_base(self, cam_id=None, readings=None, gripper_poses=None, target2cam_results=None):
        # Get Calibration Data #
        if cam_id is not None:
            readings, gripper_poses = self._readings_dict[cam_id], self._pose_dict[cam_id]
            self._curr_cam_id = cam_id

        # Get Target2Cam Transformation #
        if target2cam_results is None:
            target2cam_results = self.calculate_target_to_cam(readings)
        if target2cam_results is None:
            return None

        R_target2cam, t_target2cam, successes = target2cam_results
        gripper_poses = np.array(gripper_poses)[successes]

        # Calculate Appropriate Transformations #
        t_base2gripper = [
            -R.from_euler("xyz", pose[3:6]).inv().as_matrix() @ np.array(pose[:3]) for pose in gripper_poses
        ]
        R_base2gripper = [R.from_euler("xyz", pose[3:6]).inv().as_matrix() for pose in gripper_poses]

        # Perform Calibration #
        rmat, pos = cv2.calibrateHandEye(
            R_gripper2base=R_base2gripper,
            t_gripper2base=t_base2gripper,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=4,
        )

        # Return Pose #
        pos = pos.flatten()
        angle = R.from_matrix(rmat).as_euler("xyz")
        pose = np.concatenate([pos, angle])

        return pose

    def _calibrate_gripper_to_target(self, cam_id=None, readings=None, gripper_poses=None, target2cam_results=None):
        # Get Calibration Data #
        if cam_id is not None:
            readings, gripper_poses = self._readings_dict[cam_id], self._pose_dict[cam_id]
            self._curr_cam_id = cam_id

        # Get Target2Cam Transformation #
        if target2cam_results is None:
            target2cam_results = self.calculate_target_to_cam(readings)
        if target2cam_results is None:
            return None

        R_target2cam, t_target2cam, successes = target2cam_results
        gripper_poses = np.array(gripper_poses)[successes]

        # Calculate Appropriate Transformations #
        t_base2gripper = [
            -R.from_euler("xyz", pose[3:6]).inv().as_matrix() @ np.array(pose[:3]) for pose in gripper_poses
        ]
        R_base2gripper = [R.from_euler("xyz", pose[3:6]).inv().as_matrix() for pose in gripper_poses]

        # Perform Calibration #
        rmat, pos = cv2.calibrateHandEye(
            R_gripper2base=R_target2cam,
            t_gripper2base=t_target2cam,
            R_target2cam=R_base2gripper,
            t_target2cam=t_base2gripper,
            method=4,
        )

        # Return Pose #
        pos = pos.flatten()
        angle = R.from_matrix(rmat).as_euler("xyz")
        pose = np.concatenate([pos, angle])

        return pose

    def _calculate_gripper_to_base(self, train_readings, train_gripper_poses, eval_readings=None):
        if eval_readings is None:
            eval_readings = train_readings

        # Get Eval Target2Cam Transformations #
        eval_results = self.calculate_target_to_cam(eval_readings, train=False)
        if eval_results is None:
            return None
        eval_R_target2cam, eval_t_target2cam, eval_successes = eval_results
        rmats, tvecs = [], []

        # Get Train Target2Cam Transformations #
        train_results = self.calculate_target_to_cam(train_readings)
        if train_results is None:
            return None

        # Use Training Data For Calibrations #
        gripper2target = self._calibrate_gripper_to_target(
            gripper_poses=train_gripper_poses, target2cam_results=train_results
        )
        R_gripper2target = R.from_euler("xyz", gripper2target[3:]).as_matrix()
        t_gripper2target = np.array(gripper2target[:3])

        cam2base = self._calibrate_cam_to_base(gripper_poses=train_gripper_poses, target2cam_results=train_results)
        R_cam2base = R.from_euler("xyz", cam2base[3:]).as_matrix()
        t_cam2base = np.array(cam2base[:3])

        # Calculate Gripper2Base #
        for i in range(len(eval_R_target2cam)):
            R_gripper2cam = eval_R_target2cam[i] @ R_gripper2target
            t_gripper2cam = eval_R_target2cam[i] @ t_gripper2target + eval_t_target2cam[i]

            R_gripper2base = R_cam2base @ R_gripper2cam
            t_gripper2base = R_cam2base @ t_gripper2cam + t_cam2base

            rmats.append(R_gripper2base)
            tvecs.append(t_gripper2base)

        # Return Poses #
        eulers = np.array([R.from_matrix(rmat).as_euler("xyz") for rmat in rmats])
        eval_poses = np.concatenate([np.array(tvecs), eulers], axis=1)

        return eval_poses, eval_successes

    def is_calibration_accurate(self, cam_id):
        # Set Camera #
        self._curr_cam_id = cam_id

        # Split Into Train / Test #
        readings = self._readings_dict[cam_id]
        if len(readings) == 0:
            return False
        poses = np.array(self._pose_dict[cam_id])
        ind = np.random.choice(len(readings), size=len(readings), replace=False)
        num_train = int(len(readings) * self.train_percentage)

        train_ind, test_ind = ind[:num_train], ind[num_train:]
        train_poses, test_poses = poses[train_ind], poses[test_ind]
        train_readings = [readings[i] for i in train_ind]
        test_readings = [readings[i] for i in test_ind]

        # Calculate Approximate Gripper2Base Transformations #
        results = self._calculate_gripper_to_base(train_readings, train_poses, eval_readings=test_readings)
        if results is None:
            return False
        approx_poses, successes = results
        test_poses = np.array(test_poses)[successes]

        # Calculate Per Dimension Error #
        pose_error = np.array([pose_diff(pose, approx_pose) for pose, approx_pose in zip(test_poses, approx_poses)])
        lin_error = np.linalg.norm(pose_error[:, :3], axis=0) ** 2 / pose_error.shape[0]
        rot_error = np.linalg.norm(pose_error[:, 3:6], axis=0) ** 2 / pose_error.shape[0]

        # Check Calibration Error #
        lin_success = np.all(lin_error < self.lin_error_threshold)
        rot_success = np.all(rot_error < self.rot_error_threshold)

        # print('Pose Std: ', poses.std(axis=0))
        # print('Lin Error: ', lin_error)
        # print('Rot Error: ', rot_error)

        return lin_success and rot_success


class HandCameraCalibrator(CharucoDetector):
    def __init__(self, camera, lin_error_threshold=1e-3, rot_error_threshold=1e-2, train_percentage=0.7, **kwargs):
        self.lin_error_threshold = lin_error_threshold
        self.rot_error_threshold = rot_error_threshold
        self.train_percentage = train_percentage
        super().__init__(camera, **kwargs)

    def calibrate(self, cam_id):
        return self._calibrate_cam_to_gripper(cam_id=cam_id)

    def _calibrate_cam_to_gripper(self, cam_id=None, readings=None, gripper_poses=None, target2cam_results=None):
        # Get Calibration Data #
        if cam_id is not None:
            readings, gripper_poses = self._readings_dict[cam_id], self._pose_dict[cam_id]
            self._curr_cam_id = cam_id

        # Get Target2Cam Transformation #
        if target2cam_results is None:
            target2cam_results = self.calculate_target_to_cam(readings)
        if target2cam_results is None:
            return None

        R_target2cam, t_target2cam, successes = target2cam_results
        gripper_poses = np.array(gripper_poses)[successes]

        # Calculate Appropriate Transformations #
        t_gripper2base = [np.array(pose[:3]) for pose in gripper_poses]
        R_gripper2base = [R.from_euler("xyz", pose[3:6]).as_matrix() for pose in gripper_poses]

        # Perform Calibration #
        rmat, pos = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=4,
        )

        # Return Pose #
        pos = pos.flatten()
        angle = R.from_matrix(rmat).as_euler("xyz")
        pose = np.concatenate([pos, angle])

        return pose

    def _calibrate_base_to_target(self, cam_id=None, readings=None, gripper_poses=None, target2cam_results=None):
        # Get Calibration Data #
        if cam_id is not None:
            readings, gripper_poses = self._readings_dict[cam_id], self._pose_dict[cam_id]
            self._curr_cam_id = cam_id

        # Get Target2Cam Transformation #
        if target2cam_results is None:
            target2cam_results = self.calculate_target_to_cam(readings)
        if target2cam_results is None:
            return None

        R_target2cam, t_target2cam, successes = target2cam_results
        gripper_poses = np.array(gripper_poses)[successes]

        # Calculate Appropriate Transformations #
        t_gripper2base = [np.array(pose[:3]) for pose in gripper_poses]
        R_gripper2base = [R.from_euler("xyz", pose[3:6]).as_matrix() for pose in gripper_poses]

        # Perform Calibration #
        rmat, pos = cv2.calibrateHandEye(
            R_gripper2base=R_target2cam,
            t_gripper2base=t_target2cam,
            R_target2cam=R_gripper2base,
            t_target2cam=t_gripper2base,
            method=4,
        )

        # Return Pose #
        pos = pos.flatten()
        angle = R.from_matrix(rmat).as_euler("xyz")
        pose = np.concatenate([pos, angle])

        return pose

    def _calculate_gripper_to_base(self, train_readings, train_gripper_poses, eval_readings=None):
        if eval_readings is None:
            eval_readings = train_readings

        # Get Eval Target2Cam Transformations #
        eval_results = self.calculate_target_to_cam(eval_readings, train=False)
        if eval_results is None:
            return None
        eval_R_target2cam, eval_t_target2cam, eval_successes = eval_results
        rmats, tvecs = [], []

        # Get Train Target2Cam Transformations #
        train_results = self.calculate_target_to_cam(train_readings)
        if train_results is None:
            return None

        # Use Training Data For Calibrations #
        base2target = self._calibrate_base_to_target(gripper_poses=train_gripper_poses, target2cam_results=train_results)
        R_base2target = R.from_euler("xyz", base2target[3:]).as_matrix()
        t_base2target = np.array(base2target[:3])

        cam2gripper = self._calibrate_cam_to_gripper(gripper_poses=train_gripper_poses, target2cam_results=train_results)
        R_cam2gripper = R.from_euler("xyz", cam2gripper[3:]).as_matrix()
        t_cam2gripper = np.array(cam2gripper[:3])

        # Calculate Gripper2Base #
        for i in range(len(eval_R_target2cam)):
            R_base2cam = eval_R_target2cam[i] @ R_base2target
            t_base2cam = eval_R_target2cam[i] @ t_base2target + eval_t_target2cam[i]

            R_base2gripper = R_cam2gripper @ R_base2cam
            t_base2gripper = R_cam2gripper @ t_base2cam + t_cam2gripper

            R_gripper2base = R.from_matrix(R_base2gripper).inv().as_matrix()
            t_gripper2base = -R_gripper2base @ t_base2gripper

            rmats.append(R_gripper2base)
            tvecs.append(t_gripper2base)

        # Return Poses #
        eulers = np.array([R.from_matrix(rmat).as_euler("xyz") for rmat in rmats])
        eval_poses = np.concatenate([np.array(tvecs), eulers], axis=1)

        return eval_poses, eval_successes

    def is_calibration_accurate(self, cam_id):
        # Set Camera #
        self._curr_cam_id = cam_id

        # Split Into Train / Test #
        readings = self._readings_dict[cam_id]
        if len(readings) == 0:
            return False
        poses = np.array(self._pose_dict[cam_id])
        ind = np.random.choice(len(readings), size=len(readings), replace=False)
        num_train = int(len(readings) * self.train_percentage)

        train_ind, test_ind = ind[:num_train], ind[num_train:]
        train_poses, test_poses = poses[train_ind], poses[test_ind]
        train_readings = [readings[i] for i in train_ind]
        test_readings = [readings[i] for i in test_ind]

        # Calculate Approximate Gripper2Base Transformations #
        results = self._calculate_gripper_to_base(train_readings, train_poses, eval_readings=test_readings)
        if results is None:
            return False
        approx_poses, successes = results
        test_poses = np.array(test_poses)[successes]

        # Calculate Per Dimension Error #
        pose_error = np.array([pose_diff(pose, approx_pose) for pose, approx_pose in zip(test_poses, approx_poses)])
        lin_error = np.linalg.norm(pose_error[:, :3], axis=0) ** 2 / pose_error.shape[0]
        rot_error = np.linalg.norm(pose_error[:, 3:6], axis=0) ** 2 / pose_error.shape[0]

        # Check Calibration Error #
        lin_success = np.all(lin_error < self.lin_error_threshold)
        rot_success = np.all(rot_error < self.rot_error_threshold)

        # print('Pose Std: ', poses.std(axis=0))
        # print('Lin Error: ', lin_error)
        # print('Rot Error: ', rot_error)

        return lin_success and rot_success


def calibrate_camera(
    env,
    camera_id,
    controller,
    step_size=0.01,
    pause_time=0.5,
    image_freq=10,
    obs_pointer=None,
    wait_for_controller=False,
    reset_robot=True,
):
    """Returns true if calibration was successful, otherwise returns False
    3rd Person Calibration Instructions: Press A when board in aligned with the camera from 1 foot away.
    Hand Calibration Instructions: Press A when the hand camera is aligned with the board from 1 foot away."""

    if obs_pointer is not None:
        assert isinstance(obs_pointer, dict)

    # Get Camera + Set Calibration Mode #
    camera = env.camera_reader.get_camera(camera_id)
    env.camera_reader.set_calibration_mode(camera_id)
    assert pause_time > (camera.latency / 1000)

    # Select Proper Calibration Procedure #
    hand_camera = camera.serial_number == hand_camera_id
    intrinsics_dict = camera.get_intrinsics()
    if hand_camera:
        calibrator = HandCameraCalibrator(intrinsics_dict)
    else:
        calibrator = ThirdPersonCameraCalibrator(intrinsics_dict)

    if reset_robot:
        env.reset()
    controller.reset_state()

    print("Move the gripper and board to a clear starting position, press A when ready")
    while True:
        # Collect Controller Info #
        controller_info = controller.get_info()
        start_time = time.time()

        # Get Observation #
        state, _ = env.get_state()
        cam_obs, _ = env.read_cameras()



        for full_cam_id in cam_obs["image"]:
            if camera_id not in full_cam_id:
                continue
            cam_obs["image"][full_cam_id] = calibrator.augment_image(full_cam_id, cam_obs["image"][full_cam_id])
        if obs_pointer is not None:
            obs_pointer.update(cam_obs)

        # Get Action #
        action = controller.forward({"robot_state": state})
        # action[-1] = 0 # Keep gripper open
        # print("action: ", action)

        # Regularize Control Frequency #
        comp_time = time.time() - start_time
        sleep_left = (1 / env.control_hz) - comp_time
        if sleep_left > 0:
            time.sleep(sleep_left)

        # Step Environment #
        skip_step = wait_for_controller and (not controller_info["movement_enabled"])
        if not skip_step:
            env.step(action)

        # Check Termination #
        start_calibration = controller_info["success"]
        end_calibration = controller_info["failure"]

        # Close Files And Return #
        if start_calibration:
            break
        if end_calibration:
            return False

    # Collect Data #
    time.time()
    pose_origin = state["cartesian_position"]
    i = 0

    print("Starting calibration...")
    while True:
        # Check For Termination #
        controller_info = controller.get_info()
        if controller_info["failure"]:
            print("Calibration cancelled!")
            return False

        # Start #
        start_time = time.time()
        take_picture = (i % image_freq) == 0

        # Collect Observations #
        if take_picture:
            time.sleep(pause_time)
        state, _ = env.get_state()
        cam_obs, _ = env.read_cameras()

        # Add Sample + Augment Images #
        for full_cam_id in cam_obs["image"]:
            if camera_id not in full_cam_id:
                continue
            if take_picture:
                img = deepcopy(cam_obs["image"][full_cam_id])
                pose = state["cartesian_position"].copy()
                calibrator.add_sample(full_cam_id, img, pose)
            cam_obs["image"][full_cam_id] = calibrator.augment_image(full_cam_id, cam_obs["image"][full_cam_id])

        # Update Obs Pointer #
        if obs_pointer is not None:
            obs_pointer.update(cam_obs)

        # Move To Desired Next Pose #
        calib_pose = calibration_traj(i * step_size, hand_camera=hand_camera)
        desired_pose = change_pose_frame(calib_pose, pose_origin)
        # action = np.concatenate([desired_pose, [0]])
        action = np.concatenate([desired_pose, [1]])
        env.update_robot(action, action_space="cartesian_position", blocking=False)

        # Regularize Control Frequency #
        comp_time = time.time() - start_time
        sleep_left = (1 / env.control_hz) - comp_time
        if sleep_left > 0:
            time.sleep(sleep_left)

        # Check If Cycle Complete #
        cycle_complete = (i * step_size) >= (2 * np.pi)
        if cycle_complete:
            break
        i += 1

    # SAVE INTO A JSON
    for full_cam_id in cam_obs["image"]:
        if camera_id not in full_cam_id:
            continue
        success = calibrator.is_calibration_accurate(full_cam_id)
        if not success:
            return False
        extrinsics = calibrator.calibrate(full_cam_id).tolist()
        intrinsics = intrinsics_dict[full_cam_id]["cameraMatrix"].tolist()
        update_calibration_info(full_cam_id, intrinsics, extrinsics)

    return True


def check_calibration(
    env,
    controller,
    obs_pointer=None,
    wait_for_controller=False,
    reset_robot=True,
):

    def draw_gripper(img, pos, rot, gripper, intrinsics, color=(0, 255, 255, 255)):
        gripper_open_width, gripper_close_width = 0.07, 0.02  # meters
        gripper_width = gripper * gripper_close_width + (1 - gripper) * gripper_open_width
        gripper_lines = np.array([
            [[0.0, gripper_width, 0.05], [0.0, gripper_width, 0.18]],
            [[0.0, -gripper_width, 0.05], [0.0, -gripper_width, 0.18]],
            [[0.0, gripper_width, 0.05], [0.0, -gripper_width, 0.05]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.05]],
        ])

        for (gripper_point_1, gripper_point_2) in gripper_lines:
            gripper_point_1 = pos + euler_to_rmat(rot) @ gripper_point_1
            gripper_pixel_1 = project_camera_to_image(gripper_point_1, intrinsics)
            gripper_point_2 = pos + euler_to_rmat(rot) @ gripper_point_2
            gripper_pixel_2 = project_camera_to_image(gripper_point_2, intrinsics)
            cv2.line(img, tuple(map(int, gripper_pixel_1)), tuple(map(int, gripper_pixel_2)), color, 4)
        
        gripper_point = np.array([0.0, 0.0, 0.18])
        gripper_point = pos + euler_to_rmat(rot) @ gripper_point
        gripper_pixel = project_camera_to_image(gripper_point, intrinsics)
        cv2.circle(img, tuple(map(int, gripper_pixel)), 4, color, -1)

        return img

    if reset_robot:
        env.reset()
    controller.reset_state()

    while True:
        controller_info = controller.get_info()
        start_time = time.time()

        obs = env.get_observation()
        state, _ = env.get_state()
        cam_obs, _ = env.read_cameras()

        pos, rot, gripper_pos = state["cartesian_position"][:3], state["cartesian_position"][3:], state["gripper_position"]
        for full_cam_id in cam_obs["image"]:
            extrinsics, intrinsics = obs["camera_extrinsics"][full_cam_id], obs["camera_intrinsics"][full_cam_id]
            extrinsics = np.linalg.inv(compose_transformation_matrix(extrinsics[:3], extrinsics[3:6]))
            cur_pos, cur_rot = transform_world_to_camera(pos, rot, extrinsics)
            draw_gripper(cam_obs["image"][full_cam_id], cur_pos, cur_rot, gripper_pos, intrinsics)
        if obs_pointer is not None:
            obs_pointer.update(cam_obs)

        action = controller.forward({"robot_state": state})
        comp_time = time.time() - start_time
        sleep_left = (1 / env.control_hz) - comp_time
        if sleep_left > 0:
            time.sleep(sleep_left)
        skip_step = wait_for_controller and (not controller_info["movement_enabled"])
        if not skip_step:
            env.step(action)
        
        if controller_info["success"] or controller_info["failure"]:
            break
