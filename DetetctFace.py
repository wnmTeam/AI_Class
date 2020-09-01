from tensorflow import Graph, Session
from mtcnn.mtcnn import MTCNN
from keras import backend as k


class DetectFace:

    def __init__(self):
        self.graph = Graph()
        with self.graph.as_default():
            self.session = Session(graph=self.graph)
            with self.session.as_default():
                self.detector = MTCNN()

    def processe(self, args):
        faces = []
        image = args
        k.set_session(self.session)
        with self.graph.as_default():
            results = self.detector.detect_faces(image)

        for res in results:
            x1, y1, width, height = res['box']

            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            face = image[y1:y2, x1:x2]
            faces.append(face)

        return faces
