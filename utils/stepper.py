import time
import RPi.GPIO as gpio

(direction_pin, step_pin) = (20, 21)

gpio.setmode(gpio.BCM)
gpio.setup(step_pin, gpio.OUT)
gpio.setup(direction_pin, gpio.OUT)

while True:
    
    gpio.output(direction_pin, True)
    
    for step in range(200):
        gpio.output(step_pin, True)
        time.sleep(.01)
        gpio.output(step_pin, False)
        time.sleep(.01)
        print(f'Forward step: {step}')
        
    gpio.output(direction_pin, False)
    
    for step in range(200):
        gpio.output(step_pin, True)
        time.sleep(.01)
        gpio.output(step_pin, False)
        time.sleep(.01)
        print(f'Backward step: {step}')
        
    