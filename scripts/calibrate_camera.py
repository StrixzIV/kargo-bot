import cv2
import glob
import json
import pickle
import numpy as np

chessboardSize = (9,6)
frameSize = (640,480)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

objpoints = []
imgpoints = []

images = glob.glob('../calibration_data/*.png')

for image in images:

    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

    if not ret:
        continue

    objpoints.append(objp)

    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)

    cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
    cv2.imshow('img', img)
    cv2.waitKey(5)

cv2.destroyAllWindows()

ret, cameraMatrix, distortion, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

with open('../calibration_output/calibration.pkl', 'wb') as f:
    pickle.dump((cameraMatrix, distortion), f)

with open('../calibration_output/cameraMatrix.pkl', 'wb') as f:
    pickle.dump(cameraMatrix, f)

with open('../calibration_output/distortion.pkl', 'wb') as f:
    pickle.dump(distortion, f)

print('Calibration done.')