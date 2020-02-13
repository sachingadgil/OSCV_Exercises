import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 1

#kernel = np.ones((5,5),np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

kernel = np.array([[0, 0, 1, 0, 0],
       [0, 1, 1, 1, 0],
       [1, 1, 1, 1, 1],
       [0, 1, 1, 1, 0],
       [0, 0, 1, 0, 0]], np.uint8)


if exetasknum==1:

    img = cv2.imread('j.png',0)
    erosion = cv2.erode(img,kernel,iterations = 1)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    
    plt.subplot(331),plt.imshow(img),plt.title('img')
    plt.xticks([]), plt.yticks([])
    plt.subplot(332),plt.imshow(erosion),plt.title('erosion')
    plt.xticks([]), plt.yticks([])
    plt.subplot(333),plt.imshow(dilation),plt.title('dilation')
    plt.xticks([]), plt.yticks([])

    img2 = cv2.imread('b img w n.png', 0)
    opening = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    plt.subplot(334),plt.imshow(img2),plt.title('img2')
    plt.xticks([]), plt.yticks([])
    plt.subplot(335),plt.imshow(opening),plt.title('opening')
    plt.xticks([]), plt.yticks([])
    plt.subplot(336),plt.imshow(tophat),plt.title('tophat')
    plt.xticks([]), plt.yticks([])

    img3 = cv2.imread('w img w n.png', 0)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    plt.subplot(337),plt.imshow(img3),plt.title('img3')
    plt.xticks([]), plt.yticks([])
    plt.subplot(338),plt.imshow(closing),plt.title('closing')
    plt.xticks([]), plt.yticks([])
    plt.subplot(339),plt.imshow(blackhat),plt.title('blackhat')
    plt.xticks([]), plt.yticks([])     
    
    plt.show()

if exetasknum==2:
    pass