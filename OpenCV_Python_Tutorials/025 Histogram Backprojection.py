import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 2

if exetasknum==1:
    # Algorithm in Numpy
    # did not work ###
    #roi is the object or region of object we need to find
    roi = cv2.imread('map2.jpg')
    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

    #target is the image we search in
    target = cv2.imread('map1.jpg')
    hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

    # Find the histograms using calcHist. Can be done with np.histogram2d also
    M = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    I = cv2.calcHist([hsvt],[0, 1], None, [180, 256], [0, 180, 0, 256] )

    h,s,v = cv2.split(hsvt)
    B = R[h.ravel(),s.ravel()]
    B = np.minimum(B,1)
    B = B.reshape(hsvt.shape[:2])

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(B,-1,disc,B)
    B = np.uint8(B)
    cv2.normalize(B,B,0,255,cv2.NORM_MINMAX)

    ret,thresh = cv2.threshold(B,50,255,0)

    while True:
        cv2.imshow('thresh', thresh)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

if exetasknum==2:
    # Backprojection in OpenCV
    roi = cv2.imread('map2.jpg')
    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

    target = cv2.imread('map1.jpg')
    hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

    # calculating object histogram
    roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )

    # normalize histogram and apply backprojection
    cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)

    # Now convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(dst,-1,disc,dst)

    # threshold and binary AND
    ret,thresh = cv2.threshold(dst,50,255,0)
    thresh = cv2.merge((thresh,thresh,thresh))
    res = cv2.bitwise_and(target,thresh)

    res = np.vstack((target,thresh,res))
    cv2.imwrite('map3.jpg',res)
    
    # while True:
    #     cv2.imshow('res', res)
    #     k = cv2.waitKey(1) & 0xFF
    #     if k == 27:
    #         break