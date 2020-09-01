from tensorflow import Graph, Session
from keras import backend as k
import insightface
import urllib
import urllib.request
import cv2
import numpy as np
import time




class DetectFace:

    def __init__(self):
        self.graph = Graph()
        with self.graph.as_default():
            self.session = Session(graph=self.graph)
            with self.session.as_default():
                self.model = insightface.model_zoo.get_model('retinaface_r50_v1')
                self.model.prepare(ctx_id=-1, nms=0.4)


    def processe(self, args):
        faces = []
        image = args
        k.set_session(self.session)
        with self.graph.as_default():
            bbox, _ = self.model.detect(image, threshold=0.5, scale=1.0)
        for i, person in enumerate(bbox):
            x1, y1, x2, y2 = int(person[0]), int(person[1]), int(person[2]), int(person[3])

            # x1, y1 = abs(x1), abs(y1)

            face = image[y1:y2, x1:x2]
            faces.append(face)
            # cv2.imshow(str(i), face)
            # # cv2.rectangle(img, (int(person[0]), int(person[1])), (int(person[2]), int(person[3])), (0, 0, 255), 5)

        # for res in results:
        #     x1, y1, width, height = res['box']
        #
        #     x1, y1 = abs(x1), abs(y1)
        #     x2, y2 = x1 + width, y1 + height
        #
        #     face = image[y1:y2, x1:x2]
        #     faces.append(face)

        return faces

# d= DetectFace()
# d.processe(cv2.imread("11.jpg"))
# cv2.waitKey()
