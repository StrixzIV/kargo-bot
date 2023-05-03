import cv2
import numpy as np

def process_aruco(frame: np.ndarray, aruco_dict: cv2.aruco.Dictionary, aruco_params: cv2.aruco.DetectorParameters) -> tuple[list[int], np.ndarray]:

    view_frame = frame.copy()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, aruco_dict, parameters = aruco_params)

    if not corners:
        return ([], view_frame)

    ids = ids.flatten()

    for (marker_corner, marker_id) in zip(corners, ids):

        corners = marker_corner.reshape((4, 2))

        (top_left, top_right, bottom_left, bottom_right) = corners

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

