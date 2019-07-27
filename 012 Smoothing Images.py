import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 6

if exetasknum==1:
    # 2D Convolution ( Image Filtering )
    img = cv2.imread('opencv-logo-white.png')
    kernel = np.ones((7,7),np.float32)/49
    dst = cv2.filter2D(img,-1,kernel)
    # Depth combinations
    # Input depth (src.depth())	Output depth (ddepth)
    # CV_8U	-1/CV_16S/CV_32F/CV_64F
    # CV_16U/CV_16S	-1/CV_32F/CV_64F
    # CV_32F	-1/CV_32F/CV_64F
    # CV_64F	-1/CV_64F
    # Note when ddepth=-1, the output image will have the same depth as the source.

    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    plt.show()

# Image Blurring (Image Smoothing)
if exetasknum==2:
    # Averaging
    img = cv2.imread('opencv-logo-white.png')
    blur = cv2.blur(img,(5,5))
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()

if exetasknum==3:
    # Gaussian Filtering
    # In this approach, instead of a box filter consisting of equal filter coefficients, a Gaussian kernel is used. 
    # It is done with the function, cv2.GaussianBlur(). 
    # We should specify the width and height of the kernel which should be positive and odd. 
    # We also should specify the standard deviation in the X and Y directions, sigmaX and sigmaY respectively. 
    # If only sigmaX is specified, sigmaY is taken as equal to sigmaX. 
    # If both are given as zeros, they are calculated from the kernel size. 
    # Gaussian filtering is highly effective in removing Gaussian noise from the image. 
    # If you want, you can create a Gaussian kernel with the function, cv2.getGaussianKernel().
    img = cv2.imread('opencv-logo-white.png')
    blur = cv2.GaussianBlur(img,(5,5),0)
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()

if exetasknum==4:
    # Median Filtering
    # The function cv2.medianBlur() computes the median of all the pixels under the kernel window and the central pixel is replaced with this median value. 
    # This is highly effective in removing salt-and-pepper noise.
    img = cv2.imread('opencv-logo-white.png')
    median = cv2.medianBlur(img,5)
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(median),plt.title('median')
    plt.xticks([]), plt.yticks([])
    plt.show()

if exetasknum==5:
    # Bilateral Filtering
    # The bilateral filter also uses a Gaussian filter in the space domain, but it also uses one more (multiplicative) Gaussian filter component which is a function of pixel intensity differences. 
    # The Gaussian function of space makes sure that only pixels are ‘spatial neighbors’ are considered for filtering, 
    # while the Gaussian component applied in the intensity domain (a Gaussian function of intensity differences) ensures 
    # that only those pixels with intensities similar to that of the central pixel (‘intensity neighbors’) are included to compute the blurred intensity value. 
    # As a result, this method preserves edges
    img = cv2.imread('opencv-logo-white.png')
    blur = cv2.bilateralFilter(img,5,75,75)
    # 1st param in image, 2nd is size of window, 3rd  is sigmacolor - meaning how different colrs should influence each other, 4th is how much farther a pixcel can influence
    # if 2nd is specified, it overrides 4th. 
    # 2nd parm to be 5 for realtime, 9 for offline, 100+ for cartoonish, less thn 0.1 wil have no effect
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()

if exetasknum==6:
    # trying bilateral and gussian filter in camera feed
    cap = cv2.VideoCapture(0)
    
    def make_720p():
        cap.set(3, 1280)
        cap.set(4, 720)
    make_720p()

    while(1):
        ret, frame = cap.read()

        blur = cv2.bilateralFilter(frame,7,75,75)
        median = cv2.medianBlur(frame,7)
        g = cv2.GaussianBlur(frame,(7,7),0)

        cv2.imshow('original', frame)
        cv2.imshow('Bilateral',blur)
        cv2.imshow('Gaussian',g)
        cv2.imshow('Median',median)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()