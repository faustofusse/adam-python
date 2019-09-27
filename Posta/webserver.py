from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
from imutils.object_detection import non_max_suppression
from flask import Flask
from flask import Response
import cv2
import threading
import numpy as np
import time
import imutils

app = Flask(__name__)

confidence = 0.5
outputFrame = None
lock = threading.Lock()
ct = CentroidTracker()
(H, W) = (None, None)

print("Loading models...")
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print("Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

def recognition():
	(H, W) = (None, None)
	while True:
		# read the next frame from the video stream and resize it
		frame = vs.read()
		frame = cv2.flip(frame, 1)
		frame = imutils.resize(frame, width=400)
		# if the frame dimensions are None, grab them
		if W is None or H is None:
			(H, W) = frame.shape[:2]
		# construct a blob from the frame, pass it through the network,
		# obtain our output predictions, and initialize the list of
		# bounding box rectangles
		blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (104.0, 177.0, 123.0))
		net.setInput(blob)
		detections = net.forward()
		rects = []
		# loop over the detections
		for i in range(0, detections.shape[2]):
			# filter out weak detections by ensuring the predicted
			# probability is greater than a minimum threshold
			if detections[0, 0, i, 2] > confidence:
				# compute the (x, y)-coordinates of the bounding box for
				# the object, then update the bounding box rectangles list
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				rects.append(box.astype("int"))
	
				# draw a bounding box surrounding the object so we can
				# visualize it
				(startX, startY, endX, endY) = box.astype("int")
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 2)
		# update our centroid tracker using the computed set of bounding
		# box rectangles
		objects = ct.update(rects)
		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
	
		(rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
	
		# for (x, y, w, h) in rects:
			# cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
	
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	
		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
	
		# show the output frame
		cv2.imshow("Adam", frame)

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue
				
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
			  bytearray(encodedImage) + b'\r\n')


@app.route('/')
def hello():
	return "Hello World!"


@app.route('/video')
def video():
	return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/<name>')
def hello_name(name):
	return "Hello {}!".format(name)


if __name__ == '__main__':
	t = threading.Thread(target=recognition)
	t.daemon = True
	t.start()
	app.run()
