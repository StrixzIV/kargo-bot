import cv2 
import numpy as np

def warp_image(frame: np.ndarray, points: np.float32, width: int, height: int, inverse: bool = False) -> np.ndarray:
    
    point1 = np.float32(points)
    point2 = np.float32([
        [0, 0], 
        [width, 0], 
        [0, height],
        [width, height]
    ])
    
    if inverse:
        transform_mat = cv2.getPerspectiveTransform(point2, point1)   
    
    else:
        transform_mat = cv2.getPerspectiveTransform(point1, point2)
        
    image_warped = cv2.warpPerspective(frame, transform_mat, (width, height))
    
    return image_warped


def draw_warp_points(frame: np.ndarray, points: np.float32) -> np.ndarray:
    
    for idx in range(4):
        cv2.circle(frame, (int(points[idx][0]), int(points[idx][1])), 15, (0, 0, 255), cv2.FILLED)
        
    return frame


def get_histrogram(frame: np.ndarray, min_percentage: float = 0.1, region: int = 1, display: bool = False) -> int:
        
    if region == 1:
        hist = np.sum(frame, axis = 0)
        
    else:
        hist = np.sum(frame[(frame.shape[0] // region):, :], axis = 0)
    
    max_ = np.max(hist)
    min_ = min_percentage * max_
    
    idx_array = np.where(hist >= min_)
    base_point = int(np.average(idx_array))
    
    if display:
    
        hist_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype = np.uint8)
        
        for (x, intensity) in enumerate(hist):
            cv2.line(hist_frame, (x, frame.shape[0]), (x, (frame.shape[0] - int(np.average(intensity)) // 255) // 4), (255, 0, 255), 1)
            cv2.circle(hist_frame, (base_point, frame.shape[0]), 20, (0, 255, 255), cv2.FILLED)
            
        cv2.imshow('hist', hist_frame)
    
    return base_point
    

def get_lane_curve(frame: np.ndarray) -> int:
    
    display = frame.copy()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_white, upper_white)
    masked_frame = cv2.bitwise_and(frame, frame, mask = mask)
    
    (width, height) = (mask.shape)
    
    points = np.float32([
        (((width // 2) + 120) - 80, 100),
        (((width // 2) + 120) + 80, 100),
        (((width // 2) + 120) - 80, 200),
        (((width // 2) + 120) + 80, 200)
    ])
    
    warped_image = warp_image(masked_frame, points, width, height)
    cv2.imshow('warp image', warped_image)
    cv2.imshow('warp pos', draw_warp_points(display, points))
    
    mid_point = get_histrogram(warped_image, min_percentage = 0.1, region = 4)
    curve_avg_point = get_histrogram(warped_image, min_percentage = 0.9, display = True)
    
    curve = curve_avg_point - mid_point
    
    global curve_list
    curve_list.append(curve)
    
    if len(curve_list) > curve_list_limit:
        curve_list = curve_list[1:]
        
    curve_average = int(sum(curve_list) / len(curve_list)) / 100
    
    return curve_average


if __name__ == '__main__':
    
    lower_white = np.array((0, 0, 100))
    upper_white = np.array((60, 255, 255))
    
    cam_stream = cv2.VideoCapture(0)

    curve_list = []
    curve_list_limit = 10
    
    while True:

        (is_frame, frame) = cam_stream.read()

        if not is_frame:
            break
        
        frame = cv2.resize(frame, (480, 240))
        curve_val = get_lane_curve(frame)
        
        print(f'{curve_val=} (Turn {"Right" if curve_val > 0 else "Left"})')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break