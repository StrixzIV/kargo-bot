import cv2
import uuid

cam_stream = cv2.VideoCapture(0)

while True:

    (is_frame, frame) = cam_stream.read()

    if not is_frame:
        continue

    cv2.imshow('calibration data collection', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f'../calibration_data/{uuid.uuid4()}.png', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_stream.release()
cv2.destroyAllWindows()