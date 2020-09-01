import time

import requests
from websocket import create_connection
import json

from ClientController1 import Controller
import cv2

ws = create_connection("ws://localhost:5001")

def send_recive(imageUrl, type):
    x = {'type': type, 'imageUrl': imageUrl}
    y = json.dumps(x)
    ws.send(y)
    result = json.loads(ws.recv())
    return result

def main():
    x = {'type':'client',"client":"py"}
    y = json.dumps(x)

    ws.send(y)

    controller = Controller()
    sendData = ''
    # data = send_recive( 'C:\\Users\Omar\Desktop\\smartSchoolServer\\frames\\1.jpg', "sendFrame")
    while True:
        t = time.time()
        # data = send_recive('', 'getCam')

        data = send_recive('', 'getCam')
        frame = cv2.imread(data['data'], cv2.IMREAD_COLOR)
        sendData = controller.process_frame(frame)
        data = send_recive(data['data'], 'sendFrame')
        print(sendData)
        print(data)
        print('all time ', ': ', time.time() - t)


if __name__ == '__main__':
    main()

