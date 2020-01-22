import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 1

# In short, SURF adds a lot of features to improve the speed in every step. 
# Analysis shows it is 3 times faster than SIFT while performance is comparable to SIFT. 
# SURF is good at handling images with blurring and rotation, 
# but not good at handling viewpoint change and illumination change.

if exetasknum==1:
    # SURF is also not included in OpenCV3 onwards by default as it is 'non-free' algorithm
    img = cv2.imread('fly.png',0)
    surf = cv2.SURF(400)
    kp, des = surf.detectAndCompute(img,None)
    print(len(kp))
    print(surf.hessianThreshold)
    surf.hessianThreshold=50000
    kp, des = surf.detectAndCompute(img,None)
    print(len(kp))

    img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
    plt.imshow(img2),plt.show()
    cv2.waitKey(1)

    # apply U-SURF, so that it won’t find the orientation
    surf.upright = True
    kp = surf.detect(img,None)
    img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
    plt.imshow(img2),plt.show()
    cv2.waitKey(1)

    # Finally we check the descriptor size and change it to 128 if it is only 64-dim
    print(surf.descriptorSize())
    print(surf.extended)
    surf.extended = True
    kp, des = surf.detectAndCompute(img,None)
    print(surf.descriptorSize())
    print(des.shape)