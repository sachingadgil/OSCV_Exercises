import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 1

# Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity. The contours are a useful tool for shape analysis and object detection and recognition.

# For better accuracy, use binary images. So before finding contours, apply threshold or canny edge detection.
# findContours function modifies the source image. So if you want source image even after finding contours, already store it to some other variables.
# In OpenCV, finding contours is like finding white object from black background. So remember, object to be found should be white and background should be black.

if exetasknum==1:
    im = cv2.imread('test.jpg')
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    major = cv2.__version__.split('.')[0]
    if major == '3':
        ret, contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #im.copy()->thresh
    else:
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    img = cv2.drawContours(im, contours, -1, (0,255,0), 3)
    
    while True:
        cv2.imshow('img', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    
    # img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
    # while True:
    #     cv2.imshow('All Contours', img)
    #     k = cv2.waitKey(1) & 0xFF
    #     if k == 27:
    #         break
    
    # img = cv2.drawContours(img, contours, 0, (0,255,0), 3)
    # while True:
    #     cv2.imshow('1st Contour', img)
    #     k = cv2.waitKey(1) & 0xFF
    #     if k == 27:
    #         break    