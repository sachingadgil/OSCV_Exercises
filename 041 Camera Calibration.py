import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 1
import glob

if exetasknum==1:
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    #images = glob.glob('*.jpg')
    cap = cv2.VideoCapture(0)
    def make_720p():
        cap.set(3, 1280)
        cap.set(4, 720)
    make_720p()

    while 1:
        ret0, img = cap.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,7),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,7), corners2,ret)
            cv2.imshow('img',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            cv2.waitKey(500)
        else:
            cv2.imshow('img',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            pass

    cv2.destroyAllWindows()

# The exercises to calibration, undistortion, remapping and re-projection is skipped