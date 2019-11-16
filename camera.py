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

print("Loading models...")
net = cv2.dnn.readNetFromCaffe(
    './models/deploy.prototxt', './models/res10_300x300_ssd_iter_140000.caffemodel')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


class FaceDetectionCamera(object):
    def __init__(self, socket):
        self.video = cv2.VideoCapture(0)
        self.socket = socket

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=400)
        (H, W) = (None, None)
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
                                     (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        rects = []
        for i in range(0, detections.shape[2]):
            if detections[0, 0, i, 2] > confidence:
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                rects.append(box.astype("int"))

                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
        objects = ct.update(rects)
        for (objectID, centroid) in objects.items():
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            # print('Face', objectID, '- x:', centroid[0], '- y:', centroid[1])
            self.socket.emit('values', {'face': objectID, 'x': int(centroid[0]), 'y': int(centroid[1])})
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
