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
from keras import backend as K


class FaceRecognitionProcess(multiprocessing.Process):

    def __init__(self, q_in, q_out):
        super(FaceRecognitionProcess, self).__init__(name='FaceRecognitionProcess')
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
        self.q_in = q_in
        self.q_out = q_out
        self.modelNet = load_model('facenet_keras.h5')
        print('done')


    def run(self):
        img = []
        faces = []
        with K.get_session():
            while True:
                print('reco')
                if not self.q_in.empty():
                    faces = self.q_in.get()
                    names = []
                    for face in faces:
                        face = Image.fromarray(face)
                        face = face.resize((160, 160))
                        face = asarray(face)
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
                            names.append(predict_names[0])
                        else:
                            names.append('???')
                    self.q_out.put(names)


# class MyFaceRecognitionProcess(multiprocessing.Process):
#
#     def __init__(self, q_in, q_out):
#         super(MyFaceRecognitionProcess, self).__init__(name='FaceRecognitionProcess')
#         data = load('5-celebrity-faces-dataset.npz')
#         testX_faces = data['arr_2']
#         # load face embeddings
#         data = load('5-celebrity-faces-embeddings.npz')
#         trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
#         # normalize input vectors
#         in_encoder = Normalizer(norm='l2')
#         trainX = in_encoder.transform(trainX)
#         # testX = in_encoder.transform(testX)
#         # label encode targets
#         self.out_encoder = LabelEncoder()
#         self.out_encoder.fit(trainy)
#         trainy = self.out_encoder.transform(trainy)
#         testy = self.out_encoder.transform(testy)
#         # fit model
#         self.model = SVC(kernel='linear', probability=True)
#         self.model.fit(trainX, trainy)
#         self.modelNet = load_model('facenet_keras.h5')
#         self.q_in = q_in
#         self.q_out = q_out
#
#     def run(self):
#         with K.get_session():
#             for face in faces:
#                 face = Image.fromarray(face)
#                 face = face.resize((160, 160))
#                 face = asarray(face)
#                 random_face_emb = self.get_embedding(face)
#                 # random_face_class = testy[selection]
#                 # random_face_name = out_encoder.inverse_transform([random_face_class])
#                 # prediction for the face
#                 samples = expand_dims(random_face_emb, axis=0)
#                 yhat_class = self.model.predict(samples)
#                 yhat_prob = self.model.predict_proba(samples)
#                 # get name
#                 class_index = yhat_class[0]
#                 class_probability = yhat_prob[0, class_index] * 100
#                 predict_names = self.out_encoder.inverse_transform(yhat_class)
#
#                 if (class_probability > 99.9):
#                     names.append(predict_names[0])
#                 else:
#                     names.append('???')



if __name__ == '__main__':
    qin = multiprocessing.Queue()
    qout = multiprocessing.Queue()
    qout1 = multiprocessing.Queue()
    # pro = MyFaceRecognitionProcess(q)
    # pro.start()
    pro1 = FaceRecognitionProcess(qin, qout)
    pro1.start()
    pro2 = FaceRecognitionProcess(qin, qout)
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
        time.sleep(1)
        print(qout.get())

