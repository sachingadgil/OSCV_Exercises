import numpy as np
import cv2
#from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 3

if exetasknum==1:
    #Scaling
    img = cv2.imread('opencv-logo-white.png')
    res = cv2.resize(img,None,fx=0.75, fy=0.75, interpolation = cv2.INTER_CUBIC) #float value works
    resp = cv2.resize(img,None,fx=0.75, fy=0.75, interpolation = cv2.INTER_AREA) #Inter Area is suggested for shrinking
    height, width = img.shape[:2]
    res3 = cv2.resize(img,(3*width, 3*height), interpolation = cv2.INTER_CUBIC) # Only integer value for scaling works
    cv2.imshow('res', res)
    cv2.imshow('resp', resp)
    cv2.imshow('res3', res3)
    while(1):
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

if exetasknum==2:
    #Translation
    #Translation is the shifting of object’s location. If you know the shift in (x,y) direction, let it be (t_x,t_y), you can create the transformation matrix \textbf{M} as follows:
    #M = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y  \end{bmatrix}
    img = cv2.imread('opencv-logo-white.png')
    rows, cols, channels = img.shape

    M = np.float32([[1,0,100],[0,1,50]])
    dstsamesize = cv2.warpAffine(img,M,(cols,rows))
    dstaddedsize = cv2.warpAffine(img,M,(cols+100,rows+50))
    cv2.imshow('imgs',dstsamesize)
    cv2.imshow('imga',dstaddedsize)
    while(1):
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

if exetasknum==3:
    #Rotation
    #Rotation of an image for an angle \theta is achieved by the transformation matrix of the form
    #M = \begin{bmatrix} cos\theta & -sin\theta \\ sin\theta & cos\theta   \end{bmatrix}
    #But OpenCV provides scaled rotation with adjustable center of rotation so that you can rotate at any location you prefer. Modified transformation matrix is given by
    #\begin{bmatrix} \alpha &  \beta & (1- \alpha )  \cdot center.x -  \beta \cdot center.y \\ - \beta &  \alpha &  \beta \cdot center.x + (1- \alpha )  \cdot center.y \end{bmatrix}
    #where:
    #\begin{array}{l} \alpha =  scale \cdot \cos \theta , \\ \beta =  scale \cdot \sin \theta \end{array}
    #To find this transformation matrix, OpenCV provides a function, cv2.getRotationMatrix2D. Check below example which rotates the image by 90 degree with respect to center without any scaling.
    img = cv2.imread('opencv-logo-white.png')
    rows, cols, channels = img.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    dstby90 = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow('img', img)
    cv2.imshow('imgt',dstby90)
    #cv2.imshow('imga',dstaddedsize)
    while(1):
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()