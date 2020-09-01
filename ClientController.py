import threading


from myThread import MyThread
from DetetctFace import DetectFace
from RecognitionFace import RecognitionFace


class Controller:

    def __init__(self):
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
        # frame = cv2.imread(framePath, cv2.IMREAD_COLOR)
        self.faceDetectorThread = MyThread(
            name="faceDetectorThread",
            processor=self.faceDetector,
            event=None,
            other_events=[self.face_recognition_block],
            args=frame)
        self.faceRecogThread = MyThread(
            name="faceRecognitionThread",
            processor=self.recognitionFace,
            event=self.face_recognition_block,
            other_events=[self.emotion_detection_block],
            args=self.faceDetectorThread.mydata)

        self.faceDetectorThread.start()
        self.faceRecogThread.start()
        self.emotion_detection_block.wait()
        return self.faceRecogThread.mydata

# c = Controller()
# c.process_frame('one.jpg')
