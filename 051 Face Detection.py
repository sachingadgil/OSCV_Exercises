import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 2

if exetasknum==1:
    face_cascade = cv2.CascadeClassifier('C:\\tmp\\opencv\\build\etc\\haarcascades\\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('C:\\tmp\\opencv\\build\etc\\haarcascades\\haarcascade_eye.xml')

    img = cv2.imread('clahe_1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if exetasknum==2:
    # trying on camerafeed
    face_cascade = cv2.CascadeClassifier('C:\\tmp\\opencv\\build\etc\\haarcascades\\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('C:\\tmp\\opencv\\build\etc\\haarcascades\\haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)
    
    def make_720p():
        cap.set(3, 1280)
        cap.set(4, 720)
    make_720p()

    while(1):
        ret, frame = cap.read()
        img = frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        cv2.imshow('img',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()