import time
import RPi.GPIO as gpio

gpio.setmode(gpio.BCM)

(trig, echo) = (6, 27)

gpio.setup(trig, gpio.OUT)
gpio.setup(echo, gpio.IN)

gpio.output(trig, gpio.LOW)
print('Initializing ultrasonic sensor...')

time.sleep(2)

def get_distance() -> float:
    
    gpio.output(trig, True)
 
    time.sleep(0.00001)
    gpio.output(trig, False)
 
    start = time.time()
    stop = time.time()
 
    while gpio.input(echo) == 0:
        start = time.time()
 
    while gpio.input(echo) == 1:
        stop = time.time()
 
    timedelta = stop - start
    distance = round(timedelta * 17150, 2)
    
    return distance


if __name__ == '__main__':
    
    while True:
        distance = get_distance()
        print(distance)