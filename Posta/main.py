from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from camera import FaceDetectionCamera

app = Flask(__name__)
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(FaceDetectionCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('holaa')
def handle_message(message):
    print('received message (hola): ' + message)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', debug=True)
    socketio.run(app, host='0.0.0.0', port=5000)
