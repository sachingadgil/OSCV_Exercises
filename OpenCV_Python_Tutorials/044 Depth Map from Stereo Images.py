import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 1

if exetasknum==1:
    imgL = cv2.imread('tsukuba_l.png', 0) # This has imported the image in grayscale
    imgR = cv2.imread('tsukuba_r.png', 0)

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL,imgR)

    plt.subplot(131), plt.imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)), plt.title('Left') # Thats why color conversion won't make a difference, however without this 
    plt.subplot(132), plt.imshow(disparity,'gray'), plt.title('Depth')
    plt.subplot(133), plt.imshow(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)), plt.title('Right')
    plt.show()