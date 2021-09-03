from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, send, emit
from face_detection.camera import PersonalizedCamera
from question_and_answer.model import QuestionAnswerModel
from cv2 import cv2
from flask_cors import CORS
import socketio

app = Flask(__name__)
# socketio = SocketIO(app, cors_allowed_origins="*")
socketio = socketio.Client()
socketio.connect('http://localhost:3000/')
face_detection = PersonalizedCamera(socketio)
# face_detection = None
qa_model = QuestionAnswerModel()
# qa_model = None
CORS(app)

@socketio.event
def connect():
    print("I'm connected!")

@socketio.event
def connect_error():
    print("The connection failed!")

@socketio.event
def disconnect():
    print("I'm disconnected!")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/answer', methods=['POST'])
def answer():
    answer = qa_model.interact(request.get_json())
    response = {'answer': answer[0][0]}
    return response


@app.route('/video_feed')
def video_feed():
    return Response(face_detection.feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/users')
def get_users():
    return {'success': face_detection.names}


@app.route('/users', methods=['POST'])
def post_train():
    data = request.get_json()
    name = data['name']
    count = data['count']
    return face_detection.train(name, count)


@app.route('/users/<name>', methods=['DELETE'])
def delete_user(name):
    return face_detection.delete_person(name)


@app.route('/users/follow/start', methods=['POST'])
def start_follow():
    data = request.get_json()
    name = data['name']
    return face_detection.start_follow(name)


@app.route('/users/follow/end', methods=['POST'])
def end_follow():
    return face_detection.end_follow()


@socketio.on('mensaje')
def handle_message(message):
    print('Received message (hola): ' + str(message))


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    # socketio.run(app, host='0.0.0.0', port=5000)
