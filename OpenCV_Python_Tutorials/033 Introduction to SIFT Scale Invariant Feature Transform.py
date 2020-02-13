import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 1

if exetasknum==1:
    # SIFT is not included in OpenCV3 onwards by default as it is 'non-free' algorithm
    # this needs to be installed from opencv-contrib-python
    # API is also different sift = cv2.xfeatures2d.SIFT_create()
    img = cv2.imread('clahe_2.jpg')
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT()
    kp = sift.detect(gray,None)

    img=cv2.drawKeypoints(gray,kp)
    # to create circles around points detected 
    # img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img), plt.show()