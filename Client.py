import time

import requests
from ClientController1 import Controller
import cv2


def main():
    controller = Controller()

    # defining the api-endpoint
    API_ENDPOINT = "http://192.168.1.102:1337/getPose"
    sendData = {}

    cap = cv2.VideoCapture(0)
    while True:
        # sending post request and saving response as response object
        # r = requests.post(url=API_ENDPOINT, data=sendData)

        # extracting response text
        # recData = r.json()
        # pose = r.json()

        # frame = cv2.imread(recData)
        # sendData = controller.process_frame(frame)
        t = time.time()
        _, frame = cap.read()
        print(controller.process_frame(frame))
        print('all time ', ': ', time.time() - t)


if __name__ == '__main__':
    main()
