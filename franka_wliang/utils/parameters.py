from cv2 import aruco

##### ROBOT #####
nuc_ip = '172.16.0.4'
robot_ip = '172.16.0.2'
laptop_ip = "127.0.0.1"
sudo_password = 'robotlearning'
robot_type = "panda"  # 'panda' or 'fr3'
robot_serial_number = ""

##### CAMERAS #####
hand_camera_id = '14436910'
varied_camera_1_id = '25455306'
varied_camera_2_id = '27085680'

camera_type_dict = {
    hand_camera_id: 0,
    varied_camera_1_id: 1,
    varied_camera_2_id: 2,
}
camera_type_to_string_dict = {
    0: "hand_camera",
    1: "varied_camera_1",
    2: "varied_camera_2",
}

def get_camera_type(cam_id):
    if cam_id not in camera_type_dict:
        return None
    type_int = camera_type_dict[cam_id]
    type_str = camera_type_to_string_dict[type_int]
    return type_str

##### CHARUCO BOARD #####
CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 14
CHARUCOBOARD_CHECKER_SIZE = 0.020
CHARUCOBOARD_MARKER_SIZE = 0.016
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_100)

ubuntu_pro_token = ""

##### CODE VERSION #####
code_version = "2.0"
