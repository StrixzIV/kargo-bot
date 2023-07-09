import cv2
import pickle
from utils.aruco import process_aruco, pose_estimation

aruco_type = cv2.aruco.DICT_5X5_100
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_type)
aruco_params = cv2.aruco.DetectorParameters()

cam_stream = cv2.VideoCapture(0)

with open('./calibration_output/calibration.pkl', 'rb') as f:
    (camera_matrix, distortion_coefficients) = pickle.load(f)

while True:

    (is_frame, frame) = cam_stream.read()

    if not is_frame:
        continue

    # (ids, view_frame) = process_aruco(frame, aruco_dict, aruco_params)
    view_frame = pose_estimation(frame, aruco_dict, aruco_params, camera_matrix, distortion_coefficients)

    # print(f'ArUco ID detected: {ids}')
    cv2.imshow('ArUco View', view_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_stream.release()
cv2.destroyAllWindows()