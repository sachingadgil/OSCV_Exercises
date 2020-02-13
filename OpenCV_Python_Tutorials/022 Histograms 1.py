import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 5

if exetasknum==1:
    # 1. Histogram Calculation in OpenCV
    img = cv2.imread('home.jpg',0)
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.plot(hist)
    plt.show()

if exetasknum==2:
    # 2. Histogram Calculation in Numpy
    # OpenCV function is more faster than (around 40X) than np.histogram(). So stick with OpenCV function
    img = cv2.imread('home.jpg',0)
    hist,bins = np.histogram(img.ravel(),256,[0,256])
    plt.plot(hist)
    plt.show()

if exetasknum==3:
    # Plotting Histograms Using Matplotlib
    # greyscale
    img = cv2.imread('home.jpg',0)
    plt.hist(img.ravel(),256,[0,256]); plt.show()

if exetasknum==4:
    # Plotting Histograms Using Matplotlib
    # BGR
    img = cv2.imread('histogimg.jpg')
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

if exetasknum==5:
    # application of mask
    img = cv2.imread('home.jpg',0)

    # create a mask
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[25:425, 450:750] = 255
    masked_img = cv2.bitwise_and(img,img,mask = mask)

    # Calculate histogram with mask and without mask
    # Check third argument for mask
    hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
    hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])

    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.imshow(mask,'gray')
    plt.subplot(223), plt.imshow(masked_img, 'gray')
    plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
    plt.xlim([0,256])

    plt.show()

