import cv2
from utils.aruco import process_aruco

aruco_type = cv2.aruco.DICT_5X5_100
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_type)
aruco_params = cv2.aruco.DetectorParameters()

cam_stream = cv2.VideoCapture(0)

while True:

    (is_frame, frame) = cam_stream.read()

    if not is_frame:
        continue

    (ids, view_frame) = process_aruco(frame, aruco_dict, aruco_params)

    print(f'ArUco ID detected: {ids}')
    cv2.imshow('ArUco View', view_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_stream.release()
cv2.destroyAllWindows()