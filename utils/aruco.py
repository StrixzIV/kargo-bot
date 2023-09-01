import cv2
import math
import numpy as np

flip_mat = np.zeros((3, 3), dtype = np.float32)

flip_mat[0, 0] = 1.0
flip_mat[1, 1] = -1.0
flip_mat[2, 2] = -1.0

def is_rotation_matrix(matrix: np.matrix) -> bool:
    
    transponsed = np.transpose(matrix)
    excepted = np.dot(transponsed, matrix)

    identity_mat = np.identity(3, dtype = matrix.dtype)
    norm = np.linalg.norm(identity_mat - excepted)
    
    return norm < 1e-6


def rotation_matrix_to_euler(rotation_matrix: np.matrix) -> np.ndarray:

    assert is_rotation_matrix(rotation_matrix)

    sy = math.sqrt((rotation_matrix[0, 0] ** 2) + (rotation_matrix[0, 1] ** 2))

    singular = sy < 1e-6

    if singular:
        return np.array([
            math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]),
            math.atan2(-rotation_matrix[2, 0], sy),
            0
        ])
    
    return np.array([
        math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2]),
        math.atan2(-rotation_matrix[2, 0], sy),
        math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    ])


def process_aruco(frame: np.ndarray, aruco_dict: cv2.aruco.Dictionary, aruco_params: cv2.aruco.DetectorParameters) -> tuple[list[int], np.ndarray]:

    view_frame = frame.copy()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, aruco_dict, parameters = aruco_params)

    if not corners:
        return ([], view_frame)

    ids = ids.flatten()

    for (marker_corner, marker_id) in zip(corners, ids):

        corners = marker_corner.reshape((4, 2))

        (top_left, bottom_left, top_right, bottom_right) = corners

        top_right = (int(top_right[0]), int(top_right[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
        top_left = (int(top_left[0]), int(top_left[1]))

        cv2.line(view_frame, top_left, top_right, (0, 255, 0), 2)
        cv2.line(view_frame, top_right, bottom_right, (0, 255, 0), 2)
        cv2.line(view_frame, bottom_right, bottom_left, (0, 255, 0), 2)
        cv2.line(view_frame, bottom_left, top_left, (0, 255, 0), 2)

        c_x = int((top_left[0] + bottom_right[0]) / 2.0)
        c_y = int((top_left[1] + bottom_right[1]) / 2.0)

        cv2.circle(view_frame, (c_x, c_y), 4, (0, 0, 255), -1)

        cv2.putText(view_frame, str(marker_id), (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return (ids, view_frame)


def pose_estimation(frame: np.ndarray, aruco_dict: cv2.aruco.Dictionary, aruco_params: cv2.aruco.DetectorParameters, matrix_coefficients: np.array, distortion_coefficients: np.array) -> np.ndarray:

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    (corners, ids, rejected) = cv2.aruco.detectMarkers(gray, aruco_dict, parameters = aruco_params)
    
    data = []

    if corners:
        for i in range(0, len(ids)):
            
            (rotation_vector, transformation_vector, marker_points) = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients, distortion_coefficients)
            (transformation_vector, rotation_vector) = (transformation_vector[0, 0, :], rotation_vector[0, 0, :])
            
            cv2.putText(
                frame,
                text = f'#{ids[i]} -> tvecs = x: {transformation_vector[0]:.2f}, y: {transformation_vector[1]:.2f}, z: {transformation_vector[2]:.2f}',
                org = (0, 30),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.8,
                color = (0, 255, 0),
                thickness = 2
            )
            
            rotation_matrix = np.matrix(cv2.Rodrigues(rotation_vector)[0]).T

            (roll_rad, pitch_rad, yaw_rad) = rotation_matrix_to_euler(flip_mat * rotation_matrix)
            (roll_deg, pitch_deg, yaw_deg) = (math.degrees(roll_rad), math.degrees(pitch_rad), math.degrees(yaw_rad))
            
            cv2.putText(
                frame,
                text = f'Roll: {roll_deg:.2f}deg, Pitch: {pitch_deg:.2f}deg, Yaw: {yaw_deg:.2f}deg',
                org = (0, 70),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.8,
                color = (0, 255, 0),
                thickness = 2
            )
            
            data.append({
                'id': ids[i][0],
                'tvecs': {
                    'x': round(transformation_vector[0], 2),
                    'y': round(transformation_vector[1], 2),
                    'z': round(transformation_vector[2], 2)
                },
                'rvecs': {
                    'roll': round(roll_deg, 2),
                    'pitch': round(pitch_deg, 2),
                    'yaw': round(yaw_deg, 2)
                }
            })

            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rotation_vector, transformation_vector, 0.01)

    return data