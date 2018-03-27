# -*-coding:utf8-*-#

"""
Created on 2017.11.23
@author: linchengzhang
"""

import sys
import cv2

# config
dir_path = 'path/to/your/opencv/data/haarcascades' # opencv path
model = "haarcascade_frontalface_default.xml" # model 
model_path = dir_path + "/" + model

filename = cv2.imread('ee.jpg')


def faceRe(img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    face_cascade = cv2.CascadeClassifier(model_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # cv2.putText(img, '+1s', (x, y), font, 1.2, (0, 255, 0), 1)
    cv2.imshow('Face Recognition', img)
    cv2.waitKey(0)
    print('I\'m done')


if __name__ == "__main__":
    faceRe(filename)
