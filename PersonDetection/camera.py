import numpy as np
import cv2
import imutils
from imutils.object_detection import non_max_suppression

cap = cv2.VideoCapture(0)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while(True):
    ret, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(frame, width=min(400, frame.shape[1]))
    orig = image.copy()

    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    for (xA, yA, xB, yB) in pick: 
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
	
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()