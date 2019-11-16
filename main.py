from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, send, emit
from face_detection.camera import PersonalizedCamera
from cv2 import cv2

app = Flask(__name__)
socketio = SocketIO(app)
face_detection = PersonalizedCamera(socketio)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(face_detection.feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/users')
def get_users():
    return {'success': face_detection.names}


@app.route('/users/train', methods=['POST'])
def train():
    data = request.get_json()
    name = data['name']
    count = data['count']
    face_detection.train(name, count)
    return {'success': 'Training user: ' + name}


@app.route('/users/<name>', methods=['DELETE'])
def delete_person(name):
    if (len(face_detection.names) <= 2):
        return {'error': 'There must be at least two users.'}
    face_detection.delete_person(name)
    return {'success': name + ' deleted!'}


@socketio.on('mensaje')
def handle_message(message):
    print('Received message (hola): ' + str(message))


if __name__ == '__main__':
    # app.run(host='0.0.0.0', debug=True)
    socketio.run(app, host='0.0.0.0', port=5000)
