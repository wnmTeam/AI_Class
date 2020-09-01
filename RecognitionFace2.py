from PIL import Image
from keras.models import load_model
from numpy import asarray
from numpy import expand_dims
from numpy import load
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from tensorflow import Graph, Session
from keras import backend as k

from myThread import MyThread


class RecognitionFace:
    def __init__(self):
        self.oneFaceProcessor = OneFaceRecognition()

    def processe(self, args):
        names = []
        faces = args
        threads = []

        for i, face in enumerate(faces):
            face = Image.fromarray(face)
            face = face.resize((160, 160))
            face = asarray(face)
            threads.append(MyThread(
                name='one reco thread ' + str(i),
                event=None,
                other_events=[],
                processor=self.oneFaceProcessor,
                args=face
            ))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for i in range(len(faces)):
            names.append(threads[i].mydata)
        return names


class OneFaceRecognition:
    def __init__(self):
        self.graph = Graph()
        with self.graph.as_default():
            self.session = Session(graph=self.graph)
            with self.session.as_default():
                data = load('5-celebrity-faces-dataset.npz')
                testX_faces = data['arr_2']
                # load face embeddings
                data = load('5-celebrity-faces-embeddings.npz')
                trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
                # normalize input vectors
                in_encoder = Normalizer(norm='l2')
                trainX = in_encoder.transform(trainX)
                # testX = in_encoder.transform(testX)
                # label encode targets
                self.out_encoder = LabelEncoder()
                self.out_encoder.fit(trainy)
                trainy = self.out_encoder.transform(trainy)
                testy = self.out_encoder.transform(testy)
                # fit model
                self.model = SVC(kernel='linear', probability=True)
                self.model.fit(trainX, trainy)
                self.modelNet = load_model('facenet_keras.h5')

    def processe(self, face):
        name = ''
        k.set_session(self.session)
        with self.graph.as_default():
            random_face_emb = self.get_embedding(face)
            # random_face_class = testy[selection]
            # random_face_name = out_encoder.inverse_transform([random_face_class])
            # prediction for the face
            samples = expand_dims(random_face_emb, axis=0)
            yhat_class = self.model.predict(samples)
            yhat_prob = self.model.predict_proba(samples)
            # get name
            class_index = yhat_class[0]
            class_probability = yhat_prob[0, class_index] * 100
            predict_names = self.out_encoder.inverse_transform(yhat_class)

            if (class_probability > 99.9):
                name = predict_names[0]
            else:
                name = '???'
        return name

    def get_embedding(self, face_pixels):
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        k.set_session(self.session)
        with self.graph.as_default():
            yhat = self.modelNet.predict(samples)
        return yhat[0]
