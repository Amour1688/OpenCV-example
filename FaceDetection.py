# -*-coding: utf8 -*-

"""
Created on 2017.11.23
@author: linchengzhang
"""


import cv2


# config
dir_path = 'path/to/your/opencv/data/haarcascades' # opencv path
face = "haarcascade_frontalface_default.xml" # model
eye = 'haarcascade_eye.xml'


# Face recognition
def gface(image):
    # create classifier
    face_cascade = cv2.CascadeClassifier(dir_path + '\\' + face)
    eye_cascade = cv2.CascadeClassifier(dir_path + '\\' + eye)
    # Set gray level
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # recognition
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Draw box
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # cv2.cv2.putText(image, '+1s', (x, y), font, 1.2, (0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew , eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return image


cap = cv2.VideoCapture(0) # Capture from camera


# Get height and width of video playback interface 
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)


while(cap.isOpened()):
    # Read Frame
    ret, frame = cap.read()
    if ret:
        frame = gface(frame)
        cv2.imshow('My Camera',frame)
        # quit with Q
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    else:
        break

# release
cap.release()
cv2.destroyAllWindows()
