import cv2
import time
import pickle
import RPi.GPIO as gpio

from rich import print
from utils.aruco import pose_estimation
from utils.ultrasonic import get_distance
from utils.lane_follow import get_feedback_from_lane, power_left, power_right, in1, in2, in3, in4, light_matrix

aruco_type = cv2.aruco.DICT_5X5_100
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_type)
aruco_params = cv2.aruco.DetectorParameters()

target_station = int(input('Target station: '))

cam_stream = cv2.VideoCapture(0)

with open('./calibration_output/calibration.pkl', 'rb') as f:
    (camera_matrix, distortion_coefficients) = pickle.load(f)


try:
    while True:

        light_matrix.fill((255, 255, 255))
        (is_frame, frame) = cam_stream.read()

        if not is_frame:
            continue
        
        distance = get_distance()
                    
        print(f'Distance: {distance}')
        
        if distance <= 50:
            
            power_left.ChangeDutyCycle(0)
            power_right.ChangeDutyCycle(0)
            
            light_matrix.fill((255, 0, 0))
            time.sleep(1)
            light_matrix.fill((255, 255, 255))
            
            continue
            

        detected_markers = pose_estimation(frame, aruco_dict, aruco_params, camera_matrix, distortion_coefficients)
        print(detected_markers)
        
        if not detected_markers:
            
            get_feedback_from_lane(frame, False)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                light_matrix.fill((0, 0, 0))
                gpio.cleanup()
                cam_stream.release()
                cv2.destroyAllWindows()
                break
            
            continue
            
        if detected_markers:
            
            print('Target found')
            
            for _ in range(5):
                
                if detected_markers[0]['rvecs']['pitch'] > 0:
                    
                    power_left.ChangeDutyCycle(10)
                    power_right.ChangeDutyCycle(0)
                    
                    time.sleep(0.1)
                
                elif detected_markers[0]['rvecs']['pitch'] < 0:
                    
                    power_left.ChangeDutyCycle(0)
                    power_right.ChangeDutyCycle(10)
                    
                    time.sleep(0.1)
                    
            print('Adjustment complete')
                    
            if target_station == 1:
                
                power_left.ChangeDutyCycle(50)
                power_right.ChangeDutyCycle(50)
                
                time.sleep(1.5)
                
                power_left.ChangeDutyCycle(50)
                power_right.ChangeDutyCycle(30)
                
                time.sleep(5)
                
                for _ in range(5):
                    (is_frame, feedback_frame) = cam_stream.read()
                    get_feedback_from_lane(feedback_frame)
                
                power_left.ChangeDutyCycle(0)
                power_right.ChangeDutyCycle(0)
                
                break
                
            
            elif target_station == 2:
                
                power_left.ChangeDutyCycle(50)
                power_right.ChangeDutyCycle(50)
                
                time.sleep(5)
                
                power_left.ChangeDutyCycle(0)
                power_right.ChangeDutyCycle(0)
                
                break
                
            print(f'Found: #{detected_markers[0]["id"]}')
            
            gpio.output(in1, gpio.HIGH)
            gpio.output(in2, gpio.LOW)
            gpio.output(in3, gpio.HIGH)
            gpio.output(in4, gpio.LOW)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            light_matrix.fill((0, 0, 0))
            gpio.cleanup()
            cam_stream.release()
            cv2.destroyAllWindows()
            break
        
except:
    light_matrix.fill((0, 0, 0))
    gpio.cleanup()
    cam_stream.release()
    cv2.destroyAllWindows()