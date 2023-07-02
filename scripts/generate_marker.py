import os
import cv2
import numpy as np

aruco_type = cv2.aruco.DICT_5X5_100
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_type)

try:
    id_ = int(input('Marker ID: '))
    
except (ValueError, TypeError):
    print('Error: ID must be a integer between 0 and 99')
    exit(1)

if id_ not in range(0, 100):
    print('Error: ID must be a integer between 0 and 99')
    exit(1)
    
print(f'Generating ArUCo Marker with ID #{id_}...')

marker = np.zeros((100, 100, 1), dtype = np.uint8)
cv2.aruco.drawMarker(aruco_dict, id_, 100, marker, 1)

target_path = f'./markers/marker_#{id_}.png'

cv2.imwrite(target_path, marker)

print(f'Sucessfully generated ArUCo Marker with ID #{id_}.')
print(f'The output located at {os.path.abspath(target_path)}')