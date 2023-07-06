import os

from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)

PORT = 3000

app.config['MAX_CONTENT_LENGTH'] = 1024

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