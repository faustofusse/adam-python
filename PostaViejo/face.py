from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from imutils.object_detection import non_max_suppression

confidence = 0.85

ct = CentroidTracker()
(H, W) = (None, None)

print("Loading models...")
net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print("Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# read the next frame from the video stream and resize it
	frame = vs.read()
	frame = cv2.flip(frame, 1)
	frame = imutils.resize(frame, width=400)
	# if the frame dimensions are None, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# ----- Face detection & tracking --------------------------------------------------------

	# construct a blob from the frame, pass it through the network,
	# obtain our output predictions, and initialize the list of
	# bounding box rectangles
	blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
		(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	rects = []
	# loop over the detections
	for i in range(0, detections.shape[2]):
		if detections[0, 0, i, 2] > confidence:
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			rects.append(box.astype("int"))

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
		print('Face', objectID, '- x:', centroid[0], '- y:', centroid[1])

	# ----- Person detection -------------------------------------------------------------------

	(rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# ----- Show image to user -----------------------------------------------------------------

	cv2.imshow("Adam", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

# cleanup
cv2.destroyAllWindows()
vs.stop()