import cv2
import os

images = []
path = ''
for filename in os.listdir('../../eee'):
    img = cv2.imread(os.path.join('../../eee', filename))
    if img is not None:
        img = cv2.resize(img, (350, 350), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join('../../omar1', filename), img)
