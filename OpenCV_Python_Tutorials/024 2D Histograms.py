import os

import numpy as np
from matplotlib import pyplot as plt

import cv2

os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 2

if exetasknum==1:
    img = cv2.imread('histogimg.jpg')
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    plt.plot(hist)
    plt.show()

if exetasknum==2:
    img = cv2.imread('histogimg.jpg')
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )

    plt.imshow(hist,interpolation = 'nearest')
    plt.show()

if exetasknum==3:
    pass
