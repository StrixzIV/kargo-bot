import os
import RPi.GPIO as gpio

from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)

PORT = 3000

app.config['MAX_CONTENT_LENGTH'] = 1024

(en_left, en_right) = (13, 19)
(in1, in2, in3, in4) = (25, 24, 23, 18)
(direction_pin, step_pin) = (20, 21)

gpio.setmode(gpio.BCM)

gpio.setup(in1, gpio.OUT)
gpio.setup(in2, gpio.OUT)
gpio.setup(in3, gpio.OUT)
gpio.setup(in4, gpio.OUT)

gpio.setup(en_left, gpio.OUT)
gpio.setup(en_right, gpio.OUT)

gpio.setup(step_pin, gpio.OUT)
gpio.setup(direction_pin, gpio.OUT)

power_left = gpio.PWM(en_left, 50)
power_right = gpio.PWM(en_right, 50)

power_left.start(0)
power_right.start(0)

def show_ready(port: int) -> None:

    os.system('cls' if os.name == 'nt' else 'clear')
    
    print('REST API ready!')
    print('Press ctrl+c to exit.')
    print(f'API running on: http://localhost:{port}')
    

@cross_origin()
@app.route('/control', methods = ['POST'])
def send_response():
    
    body = request.get_json()
    
    if body is None or body['action'] == '':
        return ({}, 400)
    
    return ({'action': f'You selected: {body["action"]}'}, 200)


if __name__ == '__main__':
    show_ready(PORT)
    app.run(host = '0.0.0.0', port = PORT)