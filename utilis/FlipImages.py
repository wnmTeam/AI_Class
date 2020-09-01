import cv2
import os

path = "C:\\Users\Omar\Desktop\eee\\New folder (2) - Copy"
names = os.listdir(path)
for name in names:
    imgpath = os.path.join(path, name)
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    img = cv2.flip(img,1)
    cv2.imwrite(imgpath, img)
