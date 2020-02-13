import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
import math
exetasknum = 2

if exetasknum==1:
    img = cv2.imread('dave1.jpg',0)
    lower_reso = cv2.pyrDown(img)
    higher_reso2 = cv2.pyrUp(lower_reso)


    plt.subplot(121),plt.imshow(lower_reso)
    plt.title('Lower Reso'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(higher_reso2)
    plt.title('Higher Reso2'), plt.xticks([]), plt.yticks([])

    plt.show()


if exetasknum==2:
    # This works only if size of images matches and imageHeightOrWidth^(1/layers) is an integer
    A = cv2.imread('apple.jpg')
    B = cv2.imread('orange.jpg')

    layers=5 #higher numner results in more blending

    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in range(layers):
        G = cv2.pyrDown(gpA[i])
        gpA.append(G)

    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in range(layers):
        G = cv2.pyrDown(gpB[i])
        gpB.append(G)

    # generate Laplacian Pyramid for A
    lpA = [gpA[layers-1]]
    for i in range(layers-1,0,-1):
        size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
        GE = cv2.pyrUp(gpA[i], dstsize=size)
        L = cv2.subtract(gpA[i-1],GE)
        lpA.append(L)

    # generate Laplacian Pyramid for B
    lpB = [gpB[layers-1]]
    for i in range(layers-1,0,-1):
        size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
        GE = cv2.pyrUp(gpB[i], dstsize=size)
        L = cv2.subtract(gpB[i-1],GE)
        lpB.append(L)

    # Now add left and right halves of images in each level
    LS = []
    for la,lb in zip(lpA,lpB):
        rows,cols,dpt = la.shape
        ls = np.hstack((la[:,0:(math.floor(cols/2))], lb[:,(math.ceil(cols/2)):]))
        LS.append(ls)
    # for i in range(4):
    #     print(np.shape(LS[i]))
    # now reconstruct
    ls_ = LS[0]
    for i in range(1,layers):
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.pyrUp(ls_) #, dstsize=size
        # print(np.shape(ls_))
        ls_ = cv2.add(ls_, LS[i])

    # image with direct connecting each half
    real = np.hstack((A[:,:(math.floor(cols/2))],B[:,(math.ceil(cols/2)):]))

    # while True:
    #     cv2.imshow('Pyramid_blending2.jpg',ls_)
    #     cv2.imshow('Direct_blending.jpg',real)
    #     k = cv2.waitKey(1) & 0xFF
    #     if k == 27:
    #         break

    plt.subplot(221),plt.imshow(A)
    plt.title('Apple'), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(B)
    plt.title('Orange'), plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(ls_)
    plt.title('Pyramid Blending with ' + str(layers) +' layers'), plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(real)
    plt.title('Direct Blending'), plt.xticks([]), plt.yticks([])    

    plt.show()