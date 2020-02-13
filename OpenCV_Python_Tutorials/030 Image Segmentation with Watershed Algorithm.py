import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 1

if exetasknum==1:
    img = cv2.imread('water_coins.jpg')
    orig=img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    plt.subplot(191), plt.imshow(orig), plt.title('orig')
    plt.subplot(192), plt.imshow(thresh), plt.title('thresh')
    plt.subplot(193), plt.imshow(opening), plt.title('opening')
    plt.subplot(194), plt.imshow(sure_bg), plt.title('sure_bg')
    plt.subplot(195), plt.imshow(dist_transform), plt.title('dist_transform')
    plt.subplot(196), plt.imshow(sure_fg), plt.title('sure_fg')
    plt.subplot(197), plt.imshow(unknown), plt.title('unknown')
    plt.subplot(198), plt.imshow(markers), plt.title('markers')
    plt.subplot(199), plt.imshow(img), plt.title('img')
    plt.show()