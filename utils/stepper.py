import time
import RPi.GPIO as gpio

(direction_pin, step_pin) = (20, 21)

gpio.setmode(gpio.BCM)
gpio.setup(step_pin, gpio.OUT)
gpio.setup(direction_pin, gpio.OUT)

gpio.output(direction_pin, False)

for step in range(1600):
    gpio.output(step_pin, True)
    time.sleep(.001)
    gpio.output(step_pin, False)
    time.sleep(.001)
    print(f'Forward step: {step}')
    
gpio.output(direction_pin, True)

for step in range(1600):
    gpio.output(step_pin, True)
    time.sleep(.001)
    gpio.output(step_pin, False)
    time.sleep(.001)
    print(f'Backward step: {step}')
        
    