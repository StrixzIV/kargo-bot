import time
import RPi.GPIO as gpio

(en_left, en_right) = (13, 19)
(in1, in2, in3, in4) = (25, 24, 23, 18)

gpio.setmode(gpio.BCM)

gpio.setup(in1, gpio.OUT)
gpio.setup(in2, gpio.OUT)
gpio.setup(in3, gpio.OUT)
gpio.setup(in4, gpio.OUT)

gpio.setup(en_left, gpio.OUT)
gpio.setup(en_right, gpio.OUT)

power_left = gpio.PWM(en_left, 50)
power_right = gpio.PWM(en_right, 50)

power_left.start(0)
power_right.start(0)

power_left.ChangeDutyCycle(100)
power_right.ChangeDutyCycle(100)

gpio.output(in1, gpio.HIGH)
gpio.output(in2, gpio.LOW)
gpio.output(in3, gpio.HIGH)
gpio.output(in4, gpio.LOW)


time.sleep(5)


power_left.ChangeDutyCycle(0)
power_right.ChangeDutyCycle(0)

gpio.output(in1, gpio.LOW)
gpio.output(in2, gpio.LOW)
gpio.output(in3, gpio.LOW)
gpio.output(in4, gpio.LOW)