import numpy as np
import cv2
import imutils
from imutils.object_detection import non_max_suppression

from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import argparse
import time