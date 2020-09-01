import multiprocessing
import time

import cv2
from PIL import Image
from keras.models import load_model
from numpy import asarray
from numpy import expand_dims
from numpy import load
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from tensorflow import Graph, Session
from keras import backend as K
from mtcnn import MTCNN


class FaceDetectProcess(multiprocessing.Process):

    def __init__(self, q_in, q_out):
        super(FaceDetectProcess, self).__init__(name='FaceDetectProcess')
        self.detector = MTCNN()
        self.q_in = q_in
        self.q_out = q_out

    def run(self):
        img = []
        faces = []
        with K.get_session():

            while True:
                print('detect')
                if not self.q_in.empty():
                    t = time.time()
                    faces = []
                    img = self.q_in.get()
                    results = self.detector.detect_faces(img)
                    for res in results:
                        x1, y1, width, height = res['box']

                        x1, y1 = abs(x1), abs(y1)
                        x2, y2 = x1 + width, y1 + height

                        face = img[y1:y2, x1:x2]
                        faces.append(face)
                    self.q_out.put(faces)
                    print('face detecte time ', ': ', time.time() - t)




