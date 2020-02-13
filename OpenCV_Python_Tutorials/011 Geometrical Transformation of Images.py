import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 5

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

    scale = 1
    x_rot = img.shape[1]
    y_rot = img.shape[0]
    theta = math.radians(30) #angle of rotation in radian
    alpha = scale*math.cos(theta)
    beta = scale*math.sin(theta)
    #MathM = [[alpha, beta, (1-alpha)*x_rot-beta*y_rot][-1*beta, alpha, beta*x_rot +(1-alpha)*y_rot]]
    MathM = [alpha, beta, (1-alpha)*x_rot-beta*y_rot, -1*beta, alpha, beta*x_rot +(1-alpha)*y_rot]
    npM = np.asarray(MathM)
    MathM = npM.reshape(M.shape)
    #MathMT = [[alpha, -1*beta], [beta, alpha],[(1-alpha)*x_rot-beta*y_rot, beta*x_rot +(1-alpha)*y_rot]]
    S = np.float32([[1,0,img.shape[1]],[0,1,img.shape[0]]])
    shiftedimage = cv2.warpAffine(img,S,(cols+img.shape[1],rows+img.shape[0]))
    dstbymath = cv2.warpAffine(shiftedimage, MathM, (3*cols, 3*rows))
    #cv2.imshow('img', img)
    #cv2.imshow('imgt',dstby90)
    print(M)
    print(MathM)
    #print(MathM.shape)
    cv2.imshow('shifted', shiftedimage)
    cv2.imshow('imgm',dstbymath)
    while(1):
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

if exetasknum==4:
    #Affine Transformation
    img = cv2.imread('opencv-logo-white.png')
    rows,cols,ch = img.shape

    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])

    M = cv2.getAffineTransform(pts1,pts2) # These 2 points define rotation / scaling along the two axis

    dst = cv2.warpAffine(img,M,(cols,rows))

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()

if exetasknum==5:
    #Perspective Transformation
    img = cv2.imread('opencv-logo-white.png')
    rows,cols,ch = img.shape

    pts1 = np.float32([[5,5],[170,5],[5,165],[170,165]])
    pts2 = np.float32([[0,0],[175,0],[0,170],[175,170]])
    # these 2 points provide 2 scalings for x and y axis, rotation can be achieved as well
    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(img,M,(175,170)) # This defines resultant image size

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()