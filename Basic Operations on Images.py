import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 6
img = cv2.imread('B2DBy.jpg')

if exetasknum == 1:
    
    px = img[250, 250]
    print(px)

    # accessing only blue pixel
    blue = img[250, 250, 0]
    print(blue)

if exetasknum == 2:
    #better pixel accessing and editing method

    #Accessing RED
    print(img.item(250,250,2))
    img.itemset((250,250,2), 32)
    print(img.item(250,250,2))

if exetasknum == 3:
    #Accessing Image Properties
    print(img.shape)
    print(img.size)
    print(img.shape[0], "*", img.shape[1], "*", img.shape[2],"=",img.shape[0]*img.shape[1]*img.shape[2])
    print(img.dtype)

if exetasknum == 4:
    # Image ROI
    img2 = img.copy()
    nose = img2[170:225, 170:220]
    img2[240:295, 170:220] = nose
    cv2.imshow('image',img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if exetasknum == 5:
    # Splitting and Merging Image Channels
    b,g,r = cv2.split(img) # cv2.split() is a costly operation (in terms of time), so only use it if necessary
    img = cv2.merge((b,g,r))
    img[:,:,2] = 0
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if exetasknum == 6:
    # Making Borders for Images (Padding)
    # Image is displayed with matplotlib. So RED and BLUE planes will be interchanged
    BLUE = [255,0,0]
    img1 = img.copy()

    replicate = cv2.copyMakeBorder(img1,25,25,25,25,cv2.BORDER_REPLICATE)
    reflect = cv2.copyMakeBorder(img1,25,25,25,25,cv2.BORDER_REFLECT)
    reflect101 = cv2.copyMakeBorder(img1,25,25,25,25,cv2.BORDER_REFLECT_101)
    wrap = cv2.copyMakeBorder(img1,25,25,25,25,cv2.BORDER_WRAP)
    constant= cv2.copyMakeBorder(img1,25,25,25,25,cv2.BORDER_CONSTANT,value=BLUE)

    plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
    plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
    plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
    plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
    plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
    plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

    plt.show()