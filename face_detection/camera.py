from face_detection.pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
import argparse
import imutils
from imutils import paths
import time
from cv2 import cv2
from imutils.object_detection import non_max_suppression
import pickle
import os
import shutil

models = './face_detection/models'
dataset = './dataset'
embedding_model = './face_detection/models/openface_nn4.small2.v1.t7'
recognizer_path = './face_detection/output/recognizer.pickle'
label_encoder = './face_detection/output/le.pickle'
embeddings = './face_detection/output/embeddings.pickle'

protoPath = os.path.sep.join([models, "deploy.prototxt"])
modelPath = os.path.sep.join(
    [models, "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch(embedding_model)

ct = CentroidTracker()

green = (0, 255, 0)
blue = (255, 255, 0)
yellow = (59, 235, 255)
light_blue = (244, 169, 3)


class PersonalizedCamera(object):
    def __init__(self, socket):
        # ----
        self.video = cv2.VideoCapture(0)
        self.socket = socket
        self.count = 0
        self.max_count = 0
        self.confidence = 0.5
        self.name = ''
        self.training = False
        self.following = False
        self.searching = False
        self.id = None
        # ----
        self.recognizer = pickle.loads(open(recognizer_path, "rb").read())
        self.le = pickle.loads(open(label_encoder, "rb").read())
        data = pickle.loads(open(embeddings, "rb").read())
        self.knownEmbeddings = list(data['embeddings'])
        self.knownNames = list(data['names'])
        self.names = {}
        for n in self.knownNames:
            if (n in self.names):
                self.names[n] += 1
            else:
                self.names[n] = 1
        print(self.names)

    def refresh(self):
        self.recognizer = pickle.loads(open(recognizer_path, "rb").read())
        self.le = pickle.loads(open(label_encoder, "rb").read())
        data = pickle.loads(open(embeddings, "rb").read())
        self.knownEmbeddings = list(data['embeddings'])
        self.knownNames = list(data['names'])
        self.names = {}
        for n in self.knownNames:
            if (n in self.names):
                self.names[n] += 1
            else:
                self.names[n] = 1
        print(self.names)

    def __del__(self):
        self.video.release()

    def delete_person(self, name):
        if (len(self.names) <= 2):
            return {'error': 'There must be at least two users.'}
        for i in range(0, self.knownNames.count(str(name))):
            for i in range(0, len(self.knownNames)):
                if (self.knownNames[i] == str(name)):
                    del(self.knownNames[i])
                    del(self.knownEmbeddings[i])
                    break
        data = {'embeddings': self.knownEmbeddings, 'names': self.knownNames}
        f = open(embeddings, 'wb')
        f.write(pickle.dumps(data))
        f.close()
        self.train_model()
        self.refresh()
        return {'success': name + ' deleted!'}

    def start_follow(self, name):
        if (self.training):
            return {'error': 'No se puede empezar el follow mientras se esta entrenando.'}
        self.name = name
        self.following = True
        self.searching = True
        self.max_count = 50
        self.count = 0
        return {'success': 'Started following ' + name}

    def end_follow(self):
        self.name = ''
        self.following = False
        self.id = None
        return {'success': 'Stopped following'}

    def train(self, name, count):
        if (self.following):
            return {'error': 'No se puede entrenar mientras esta el follow activado.'}
        self.name = name.lower()
        self.max_count = int(count)
        self.training = True
        try:
            os.mkdir(dataset)
            os.mkdir(dataset + '/' + self.name)
        except:
            return {'error': 'Error interno (creacion de la carpeta para el dataset).'}
            print('Error en la creacion de la carpeta')
        return {'success': 'Training user: ' + name}

    def extract_embeddings(self):
        print('Extracting embeddings')
        imagePaths = list(paths.list_images(dataset))
        total = 0
        for (i, imagePath) in enumerate(imagePaths):
            print("[INFO] processing image {}/{}".format(i + 1,
                                                         len(imagePaths)))
            name = imagePath.split(os.path.sep)[-2]
            image = cv2.imread(imagePath)
            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)
            detector.setInput(imageBlob)
            detections = detector.forward()
            if len(detections) > 0:
                i = np.argmax(detections[0, 0, :, 2])
                conf = detections[0, 0, i, 2]
                if conf > self.confidence:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    face = image[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]
                    if fW < 20 or fH < 20:
                        continue
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                     (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()
                    self.knownNames.append(name)
                    self.knownEmbeddings.append(vec.flatten())
                    total += 1
        data = {'embeddings': self.knownEmbeddings, 'names': self.knownNames}
        f = open(embeddings, 'wb')
        f.write(pickle.dumps(data))
        f.close()
        shutil.rmtree(dataset)

    def train_model(self):
        self.le = LabelEncoder()
        labels = self.le.fit_transform(self.knownNames)
        print("[INFO] training model...")
        self.recognizer = SVC(C=1.0, kernel="linear", probability=True)
        self.recognizer.fit(self.knownEmbeddings, labels)
        f = open(recognizer_path, "wb")
        f.write(pickle.dumps(self.recognizer))
        f.close()
        f = open(label_encoder, "wb")
        f.write(pickle.dumps(self.le))
        f.close()

    def get_frame_to_dataset(self):
        success, frame = self.video.read()
        frame = imutils.resize(frame, width=600)
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()
        faces = 0
        for i in range(0, detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > self.confidence:
                faces += 1
        if (self.count >= self.max_count - 5):
            cv2.putText(frame, 'Processing...', (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 0), 2)
        if (faces != 1):
            cv2.putText(frame, 'There must be one face on camera.', (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            print('Tiene que haber una cara en la camara.')
        elif (self.count >= self.max_count):
            print('Se termino la recoleccion de frames.')
            self.training = False
            self.max_count = 0
            self.count = 0
            self.extract_embeddings()
            self.train_model()
            self.refresh()
        else:
            cv2.putText(frame, 'Training...', (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 0), 2)
            cv2.imwrite('./dataset/' + self.name + '/' + self.name + '.' +
                        str(self.count) + ".jpg", frame)
            self.count += 1
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def get_frame(self):
        success, frame = self.video.read()
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()
        rects = []
        names = []
        color = None
        for i in range(0, detections.shape[2]):
            conf = detections[0, 0, i, 2]
            # print(conf)
            if conf > self.confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                if fW < 20 or fH < 20:
                    continue
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                preds = self.recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = self.le.classes_[j]
                if (proba < 0.95):
                    name = 'unknown'
                text = "{}: {}%".format(name, int(proba * 100))
                y = startY - 10 if startY - 10 > 10 else startY + 10

                if (self.following):
                    if (self.id is not None):
                        names.append(name)
                        rects.append(box.astype("int"))
                        print('Id already set ({})'.format(self.id))
                    elif (self.name == name):
                        names = [name]
                        rects = [box.astype("int")]
                        print('Trying to set id')
                else:
                    rects.append(box.astype("int"))

                if (not self.following):
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  green, 2)
                if (not self.following):
                    cv2.putText(frame, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (color if (color is not None) else green), 2)

        objects = ct.update(rects)
        aparecio = False if len(objects) > 0 else True
        i = 0
        for (objectID, centroid) in objects.items():
            if (len(rects) > i):
                (startX, startY, endX, endY) = rects[i]
            else:
                (startX, startY, endX, endY) = (0, 0, 0, 0)
            if (self.following):
                if (self.id is not None):
                    if (self.id == objectID):
                        aparecio = True
                    if (self.count == self.max_count and self.id == objectID):
                        print('Send Data')
                        color = blue
                        
                        self.socket.emit(
                            'follow', {'position': {'x': str(int((centroid[0] * 100) / w)), 'y': str(int((centroid[1] * 100) / h))},
                                       'size': {'x': str(int(((endX - startX) * 100) / w)), 'y': str(int(((endY - startY) * 100) / h))}})
                    else:
                        if (self.id == objectID and len(names) > 0 and self.name == names[i]):
                            self.count += 1
                            print(self.count)
                            color = yellow
                            cv2.putText(frame, '{}/{}'.format(self.count, self.max_count), (startX, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, yellow, 2)
                        elif (self.id == objectID):
                            self.count = 0
                else:
                    if (len(names) > 0 and self.name == names[i]):
                        self.id = objectID
                        print('Set id ({})'.format(self.id))
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (color if (color is not None) else green), 2)

            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (color if (color is not None) else green), 2)
            cv2.circle(
                frame, (centroid[0], centroid[1]), 4, (color if (color is not None) else green), -1)
            i += 1

        if (self.following and ((len(objects) == 0 and self.id is not None) or (self.count == self.max_count and not aparecio))):
            self.id = None
            self.count = 0
            print('Perdi el id.')

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def feed(self):
        while True:
            if(self.training):
                frame = self.get_frame_to_dataset()
            else:
                frame = self.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
