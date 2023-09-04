import cv2 
import numpy as np

import time
import board
import neopixel
import RPi.GPIO as gpio

(en_left, en_right) = (19, 13)
(in1, in2, in3, in4) = (25, 24, 23, 18)

(pre_error, error_sum) = (0, 0)

gpio.setmode(gpio.BCM)

gpio.setup(in1, gpio.OUT)
gpio.setup(in2, gpio.OUT)
gpio.setup(in3, gpio.OUT)
gpio.setup(in4, gpio.OUT)

gpio.output(in1, gpio.HIGH)
gpio.output(in2, gpio.LOW)
gpio.output(in3, gpio.HIGH)
gpio.output(in4, gpio.LOW)

gpio.setup(en_left, gpio.OUT)
gpio.setup(en_right, gpio.OUT)

power_left = gpio.PWM(en_left, 50)
power_right = gpio.PWM(en_right, 50)

power_left.start(0)
power_right.start(0)

light_matrix = neopixel.NeoPixel(board.D10, 16, brightness = 0.3)

kernel = np.ones((3, 3), np.float32) / 10

def ROI(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    
    mask = np.zeros_like(frame)
    match_mask_color = 255
    
    vertices = np.array([[
        (0, (height / 2) + 80),
        (width, (height / 2) + 80),
        (width, height),
        (0, height),
    ]], np.int32)
      
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(frame, mask)
    
    return masked_image



def warp_perspective(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    
    '''
        Warp selected part of image and warp it into top view image
    '''
    
    # Offset for frame ratio saving
    offset = 30    
    
    source_points = np.float32([
        [(width*0.2), int(height*0.5)], # Top-left point
        [(width*0.8), int(height*0.5)], # Top-right point
        [0, height], # Bottom-left point
        [width, height], # Bottom-right point
    ])
    
    destination_points = np.float32([
        [offset, 0], # Top-left point
        [width-2*offset, 0], # Top-right point
        [offset, height], # Bottom-left point
        [width-2*offset, height] # Bottom-right point
    ])

    matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    warped_frame = cv2.warpPerspective(frame, matrix, (width, height))

    return warped_frame


def detect_lines(frame: np.ndarray) -> any:
    line_segments = cv2.HoughLinesP(frame, 1, np.pi / 180 , 50, np.array([]), minLineLength = 140, maxLineGap = 10)
    return line_segments


def map_coordinates(frame, parameters, height):
    
    slope, intercept = parameters   # Unpack slope and intercept from the given parameters
    
    if slope == 0:      # Check whether the slope is 0
        slope = 0.1     # handle it for reducing Divisiob by Zero error
    
    y1 = height             # Point bottom of the frame
    y2 = int(height*0.72)  # Make point from middle of the frame down  
    x1 = int((y1 - intercept) / slope)  # Calculate x1 by the formula (y-intercept)/slope
    x2 = int((y2 - intercept) / slope)  # Calculate x2 by the formula (y-intercept)/slope
    
    return [[x1, y1, x2, y2]]   # Return point as array


def optimize_lines(frame: np.ndarray, lines: any, width: int, height: int) -> any:
    
    if lines is not None:
        
        lane_lines = [] # For both lines
        left_fit = []   # For left line
        right_fit = []  # For right line
        
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)    # Unpack actual line by coordinates

            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            
            if slope < 0:
                left_fit.append((slope, intercept))
                
            else:   
                right_fit.append((slope, intercept))

        if len(left_fit) > 0:       # Here we ckeck whether fit for the left line is valid
            left_fit_average = np.average(left_fit, axis=0)     # Averaging fits for the left line
            lane_lines.append(map_coordinates(frame, left_fit_average, height = height)) # Add result of mapped points to the list lane_lines
            
        if len(right_fit) > 0:       # Here we ckeck whether fit for the right line is valid
            right_fit_average = np.average(right_fit, axis=0)   # Averaging fits for the right line
            lane_lines.append(map_coordinates(frame, right_fit_average, height = height))    # Add result of mapped points to the list lane_lines
        
    return lane_lines


def display_lines(frame: np.ndarray, lines: any) -> np.ndarray:
    
    mask = np.zeros_like(frame)
    
    if lines is not None:
        for line in lines:                  # Iterate through lines list
            for (x1, y1, x2, y2 )in line:     # Unpack line by coordinates
                cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    frame = cv2.addWeighted(frame, 0.8, mask, 1, 1)    
    
    return frame


def lane_to_histogram(frame: np.ndarray) -> tuple[int, int]:
   
    '''
        Convert lane image into histogram.
    '''
    histogram = np.sum(frame, axis=0)   
    
    midpoint = np.int(histogram.shape[0]/2)
    
    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint  
    
    return (left_x_base, right_x_base)


def find_center(frame: np.ndarray, lane_lines: any, width: int) -> tuple[int, int]:
    
    if len(lane_lines) != 2:
        return (int(width*1.9), int(width*1.9))

    left_x1, _, left_x2, _ = lane_lines[0][0]
    right_x1, _, right_x2, _ = lane_lines[1][0]
    
    low_mid = (right_x1 + left_x1) / 2  # Calculate the relative position of the lower middle point
    up_mid = (right_x2 + left_x2) / 2
    
    return (up_mid, low_mid)
    
   
def calculate_feedback(lane_center_point: float, left_x_base: int, right_x_base: int, base_spd: float, min_spd: float, max_spd: float, k_p: float, k_i: float, k_d: float) -> None:
    
    global pre_error, error_sum
    
    lane_center = left_x_base + (right_x_base - left_x_base) / 2
    
    deviation = (lane_center_point - lane_center) / 10
    print(lane_center, deviation)
    
    adjust = (k_p * deviation) + (k_d * (deviation - pre_error)) + (k_i * error_sum)
    
    left_spd = base_spd + adjust
    right_spd = base_spd - adjust
        
    if left_spd > max_spd: left_spd = max_spd
    elif right_spd > max_spd: right_spd = max_spd
    
    if left_spd < min_spd: left_spd = min_spd
    elif right_spd < min_spd: right_spd = min_spd
    
    power_left.ChangeDutyCycle(left_spd)
    power_right.ChangeDutyCycle(right_spd)
    
    pre_error = deviation
    error_sum += deviation
    
     
    
def get_feedback_from_lane(frame: np.ndarray, debug: bool = False, base_spd: float = 30, min_spd: float = 25, max_spd: float = 40, k_p: float = 35, k_i: float = .00015, k_d: float = 45) -> str:
    
    (height, width, _color) = frame.shape
    
    show_frame = frame.copy()
    
    frame = cv2.GaussianBlur(frame, ksize = (3, 3), sigmaX = 4)
    
    lower_red = np.array((0, 20, 20))
    upper_red = np.array((10, 255, 255))
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(frame.copy(), lower_red, upper_red)
    
    roi = ROI(red_mask, width, height)
    
    filled_roi = cv2.dilate(roi, np.ones((5, 5), dtype = np.uint8), iterations = 6)
    warped_frame = warp_perspective(filled_roi, width, height)
    
    warped_frame = cv2.morphologyEx(warped_frame, op = cv2.MORPH_CLOSE, kernel = np.ones((3, 3), dtype = np.uint8), iterations = 3)
    
    lines = detect_lines(filled_roi)
    (left_lane_base, right_lane_base) = lane_to_histogram(warped_frame)
    
    if lines is None:
        return
    
    try:
        lane_lines = optimize_lines(frame, lines, width, height)
        lane_lines_frame = display_lines(show_frame, lane_lines)
        
    except:
        return
    
    (center_top, center_bottom) = find_center(frame, lane_lines, width)
    calculate_feedback(center_bottom, left_lane_base, right_lane_base, base_spd, min_spd, max_spd, k_p, k_i, k_d)
    
    if debug:
        cv2.putText(show_frame, str(center_top), (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1) 
        cv2.imshow('ROI', filled_roi)
        cv2.imshow('Warped', warped_frame)
        cv2.imshow('Line visualize', lane_lines_frame)
        
    
if __name__ == '__main__':
    
    from ultrasonic import get_distance
    
    cam_stream = cv2.VideoCapture(0)
    
    try:
        while True:
            

                (is_frame, frame) = cam_stream.read()

                if not is_frame:
                    break
                
                distance = get_distance()
                
                print(distance)
                
                if distance <= 50:
                    
                    power_left.ChangeDutyCycle(0)
                    power_right.ChangeDutyCycle(0)
                    
                    light_matrix.fill((255, 0, 0))
                    
                    time.sleep(1)
                    
                    continue
                    
                light_matrix.fill((255, 255, 255))
                
                feedback = get_feedback_from_lane(frame, False)
                print(feedback)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    light_matrix.fill((0, 0, 0))
                    gpio.cleanup()
                    break
                
    except KeyboardInterrupt:
        light_matrix.fill((0, 0, 0))
        gpio.cleanup()