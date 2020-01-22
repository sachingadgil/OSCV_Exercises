import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 3

if exetasknum==1:
    #BackgroundSubtractorMOG
    cap = cv2.VideoCapture(0)
    def make_720p():
        cap.set(3, 1280)
        cap.set(4, 720)
    make_720p()

    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    while(1):
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)

        cv2.imshow('frame',fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if exetasknum==2:
    #BackgroundSubtractorMOG2
    cap = cv2.VideoCapture(0)
    def make_720p():
        cap.set(3, 1280)
        cap.set(4, 720)
    make_720p()

    fgbg = cv2.createBackgroundSubtractorMOG2()

    while(1):
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)

        cv2.imshow('frame',fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if exetasknum==3:
    #BackgroundSubtractorGMG
    cap = cv2.VideoCapture(0)
    def make_720p():
        cap.set(3, 1280)
        cap.set(4, 720)
    make_720p()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

    while(1):
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        cv2.imshow('frame',fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()