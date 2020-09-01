import time
from statistics import mode

import cv2
from keras.models import load_model
import numpy as np
###################################
from scipy.io import loadmat
import pandas as pd
import numpy as np
from random import shuffle
import os
import cv2
from emotion_model.src.utils2.datasets import get_labels
from tensorflow import Graph, Session
from keras import backend as k
from emotion_model.src.utils2.preprocessor import preprocess_input


class EmotionsDetection:
    def __init__(self):
        self.emotion_model_path = 'emotion_model/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
        self.emotion_labels = get_labels('fer2013')
        self.graph = Graph()
        with self.graph.as_default():
            self.session = Session(graph=self.graph)
            with self.session.as_default():
                self.emotion_classifier = load_model(self.emotion_model_path, compile=False)

        # starting lists for calculating modes
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]
        self.emotion_window = []
        self.frame_window = 10
        self.emotion_offsets = (20, 40)

    ##################################
    def processe(self, args):
        res = []
        faces = args
        for face in faces:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # try:
            face = cv2.resize(face, self.emotion_target_size)
            # except :
            #     continue
            face = preprocess_input(face, True)
            face = np.expand_dims(face, 0)
            face = np.expand_dims(face, -1)
            t1 = time.time()
            k.set_session(self.session)
            with self.graph.as_default():
                emotion_prediction = self.emotion_classifier.predict(face)
            print('time is :', time.time() - t1)
            emotion_probability = np.max(emotion_prediction)
            all_emotions = emotion_prediction
            emotion_label_arg = np.argmax(all_emotions)
            if round(all_emotions[0][6] - all_emotions[0][4], 2) >= 0.45:
                emotion_label_arg = 4
            elif round(all_emotions[0][0], 2) >= 0.2:
                emotion_label_arg = 0
            emotion_text = self.emotion_labels[emotion_label_arg]
            res.append([emotion_text, emotion_probability])
        return res


e = EmotionsDetection()
print(e.processe([cv2.imread('2.jpg', cv2.IMREAD_COLOR)]))
# while True:
#     bgr_image = video_capture.read()[1]
#     gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
#     rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
#     faces = detect_faces(face_detection, gray_image)
#
#     for face_coordinates in faces:
#
#         x1, x2, y1, y2 = apply_offsets(face_coordinates, self.emotion_offsets)
#         gray_face = gray_image[y1:y2, x1:x2]
#         try:
#             gray_face = cv2.resize(gray_face, (emotion_target_size))
#         except:
#             continue
#
#         gray_face = preprocess_input(gray_face, True)
#         gray_face = np.expand_dims(gray_face, 0)
#         gray_face = np.expand_dims(gray_face, -1)
#         t1 = time.time()
#         emotion_prediction = emotion_classifier.predict(gray_face)
#         print('time is :',time.time() - t1)
#         emotion_probability = np.max(emotion_prediction)
#         all_emotions = emotion_prediction
#         emotion_label_arg = np.argmax(all_emotions)
#         if (round(all_emotions[0][6] - all_emotions[0][4], 2) >= 0.45):
#             emotion_label_arg = 4
#         elif (round(all_emotions[0][0], 2) >= 0.2):
#             emotion_label_arg = 0
#         emotion_text = emotion_labels[emotion_label_arg]
#         # if (round(all_emotions[0][3], 2) - round(all_emotions[0][6], 2) <= 0.2):
#         #     emotion_label_arg = 6
#         emotion_window.append(emotion_text)
#
#
#         if len(emotion_window) > frame_window:
#             emotion_window.pop(0)
#         try:
#             emotion_mode = mode(emotion_window)
#         except:
#             continue
#
#         if emotion_text == 'angry':
#             color = emotion_probability * np.asarray((255, 0, 0))
#         elif emotion_text == 'sad':
#             color = emotion_probability * np.asarray((0, 0, 255))
#         elif emotion_text == 'happy':
#             color = emotion_probability * np.asarray((255, 255, 0))
#         elif emotion_text == 'surprise':
#             color = emotion_probability * np.asarray((0, 255, 255))
#         else:
#             color = emotion_probability * np.asarray((0, 255, 0))
#
#         color = color.astype(int)
#         color = color.tolist()
#
#         draw_bounding_box(face_coordinates, rgb_image, color)
#         draw_text(face_coordinates, rgb_image, emotion_mode+' '+ str(round(emotion_probability,2)),
#                   color, 0, -45, 1, 1)
#
#     bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
#     cv2.imshow('window_frame', bgr_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# video_capture.release()
# cv2.destroyAllWindows()
