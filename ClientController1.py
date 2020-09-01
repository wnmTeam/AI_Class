import threading
import time

import cv2

from myThread import MyThread
from DetetctFace1 import DetectFace
from RecognitionFace1 import RecognitionFace
from


class Controller:

    def __init__(self):
        self.clock_block = threading.Event()

        self.person_detection_block = threading.Event()
        self.face_detection_block = threading.Event()
        self.pose_detection_block = threading.Event()
        self.emotion_detection_block = threading.Event()
        self.face_direction_block = threading.Event()
        self.face_recognition_block = threading.Event()

        self.students = []
        self.faces = []
        self.poses = []
        self.names = []
        self.emo = []
        self.face_dir = []

        self.faceDetector = DetectFace()
        self.recognitionFace = RecognitionFace()

    def process_frame(self, frame):
        # clear all events
        self.person_detection_block.clear()
        self.face_detection_block.clear()
        self.pose_detection_block.clear()
        self.emotion_detection_block.clear()
        self.face_direction_block.clear()
        self.face_recognition_block.clear()

        # frame = cv2.imread(framePath, cv2.IMREAD_COLOR)

        # init all threads
        self.faceDetectorThread = MyThread(
            name="faceDetectorThread",
            processor=self.faceDetector,
            other_events=[self.face_detection_block],
            args=frame)
        self.faceRecogThread = MyThread(
            name="faceRecognitionThread",
            other_events=[self.face_recognition_block],
            processor=self.recognitionFace,
            args=self.faces)

        # start all threads
        self.faceDetectorThread.start()
        self.faceRecogThread.start()

        # waiting all events
        self.face_detection_block.wait()
        self.face_recognition_block.wait()

        # take res from threads
        self.faces = self.faceDetectorThread.mydata
        self.names = self.faceRecogThread.mydata

        # return last res
        return self.names

# c = Controller()
# cap = cv2.VideoCapture(0)
# while True:
#     t = time.time()
#     _, frame = cap.read()
#     print(c.process_frame(frame))
#     print('all time ', ': ', time.time() - t)
