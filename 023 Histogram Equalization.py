import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 7

if exetasknum==1:
    img = cv2.imread('wiki.jpg',0)

    hist,bins = np.histogram(img.flatten(),256,[0,256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

if exetasknum==2:
    img = cv2.imread('wiki.jpg',0)

    hist = cv2.calcHist([img],[0],None,[256],[0,256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

if exetasknum==3:
    img = cv2.imread('wiki.jpg',0)

    hist,bins = np.histogram(img.flatten(),256,[0,256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[img]

    plt.subplot(121), plt.imshow(img2, 'gray')
    plt.subplot(122), plt.plot(cdf_normalized, color = 'b')
    plt.hist(img2.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

if exetasknum==4:
    img = cv2.imread('wiki.jpg',0)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img,equ)) #stacking images side-by-side
    while True:
        cv2.imshow('res', res)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

if exetasknum==5:
    img = cv2.imread('wiki.jpg',0)

    hist,bins = np.histogram(img.flatten(),256,[0,256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    cdf_m = np.ma.masked_equal(cdf,0)
    # trying to changing the end points to halfway - produced cartoonish image
    cdf_m = (cdf_m - ((cdf_m.min()+0)/2))*((255+cdf_m.max())/2)/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[img]

    plt.subplot(121), plt.imshow(img2, 'gray')
    plt.subplot(122), plt.plot(cdf_normalized, color = 'b')
    plt.hist(img2.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

if exetasknum==6:
    img = cv2.imread('wiki.jpg',0)

    hist,bins = np.histogram(img.flatten(),256,[0,256])

    cdf = hist.cumsum()
    # another failed attempt
    cdf_normalized = cdf * ((255-hist.max())/2)/ cdf.max()

    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - ((cdf_m.min())/2))*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[img]

    plt.subplot(121), plt.imshow(img2, 'gray')
    plt.subplot(122), plt.plot(cdf_normalized, color = 'b')
    plt.hist(img2.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

if exetasknum==7:
    img = cv2.imread('clahe_2.jpg',0)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)

    while True:
        cv2.imshow('clahe', cl1)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break