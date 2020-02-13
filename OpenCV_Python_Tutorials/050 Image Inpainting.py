import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 1

if exetasknum==1:
    # Image Inpainting
    # Skipping this excercise as did not want to use readymade mask

    img = cv2.imread('messi_2.jpg')
    mask = cv2.imread('mask2.png',0)

    dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)

    cv2.imshow('dst',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()