import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 2

if exetasknum==1:
    img = cv2.imread('dave1.jpg',0)
    edges = cv2.Canny(img,100,200)

    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()


if exetasknum==2:
    # trying out canny on camera feed
    cap = cv2.VideoCapture(0)
    
    def make_720p():
        cap.set(3, 1280)
        cap.set(4, 720)
    make_720p()

    while(1):
        ret, frame = cap.read()

        edges = cv2.Canny(frame,150,225)

        cv2.imshow('original', frame)
        cv2.imshow('Canny',edges)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()