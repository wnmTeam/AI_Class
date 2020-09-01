# # import cv2
# #
# # # d = DetectFace()
# # # img = cv2.imread('obama.jpg', cv2.IMREAD_COLOR)
# # # img = [img]
# # # print(d.processe(img))
# # from myProject.myThread import MyThread
# #
# #
# # class y:
# #
# #     def yy(self):
# #
# #         print('yy')
# #
# # class x:
# #
# #     def __init__(self):
# #         self.yy = y()
# #
# #     def processe(self, f):
# #         self.yy.yy()
# #
# #
# # xx = x()
# #
# #
# # def processe():
# #     img = cv2.imread('../../obama.jpg', cv2.IMREAD_COLOR)
# #     # print(mtcnn.detect_faces(img))
# #
# # xx = x()
# # my = MyThread(
# #     name='omar',
# #     event=None,
# #     other_events=[],
# #     processor=xx,
# #     args=[None]
# # )
# # my.start()
#
#
# import multiprocessing
# import threading
# import os
# import time
# from concurrent.futures.thread import ThreadPoolExecutor
# from myPocesse import MyProcesse
# from DetetctFace import DetectFace
#
#
# def square(n):
#
#     return (n * n)
#
#
# class omar:
#     def processe(self, x):
#         print('omar1')
#
# class processe(multiprocessing.Process):
#     def __init__(self):
#         multiprocessing.Process.__init__(self)
#
#     def run(self):
#         print('omar')
#
#
# det = DetectFace()
#
# if __name__ == "__main__":
#     # # input list
#     # mylist = []
#     # for i in range(10000000):
#     #     mylist.append(5)
#     #
#     # # creating a pool object++++++++++++++++++
#     # p = multiprocessing.Pool(4)
#     #
#     # # t = time.time()
#     # # with ThreadPoolExecutor(max_workers=10) as executor:
#     # #     results = executor.map(square, mylist)
#     # # print(time.time() - t)
#     # # map list to target function
#     # t= time.time()
#     # result = p.map(square, mylist)
#     # # for i, j in enumerate(mylist):
#     # #     mylist[i] = square(j)
#     # print(time.time() - t)
#
#
#
#
#
#
#
#     o = omar()
#     pr = MyProcesse(
#         name='omar',
#         processe=det.processe,
#     )
#
#     pr.start()

from mtcnn import MTCNN
import cv2

det = MTCNN()
img = cv2.imread('C:\\Users\Omar\Desktop\\smartSchoolServer\\frames\\4.png', cv2.IMREAD_COLOR)
print(det.detect_faces(img))