import time

import cv2
import multiprocessing

from FaceDetectionProcess import FaceDetectProcess
from FaceRecognitionProcess import FaceRecognitionProcess

if __name__ == '__main__':
    qin = multiprocessing.Queue()
    qout = multiprocessing.Queue()
    qout1 = multiprocessing.Queue()
    # pro = MyFaceRecognitionProcess(q)
    # pro.start()
    # pro1 = FaceDetectProcess(qin, qout)
    # pro1.start()
    pro2 = FaceRecognitionProcess(qout, qout1)
    pro2.start()

    cap = cv2.VideoCapture(0)
    while True:
        # # sending post request and saving response as response object
        # r = requests.post(url=API_ENDPOINT, data=sendData)
        #
        # # extracting response text
        # recData = r.json()
        # pose = r.json()
        #
        # frame = cv2.imread(recData)
        # sendData = controller.process_frame(frame, pose)

        _, frame = cap.read()
        qin.put(frame)
        time.sleep(2)
        print(qout1.get())
