import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 2

# FAST is based on machine learning, trained on images in simiar domain
# It is several times faster than other existing corner detectors.
# But it is not robust to high levels of noise. It is dependant on a threshold.

if exetasknum==1:
    img = cv2.imread('simple.jpg',0)
    fast = cv2.FastFeatureDetector()
    # find and draw the keypoints
    kp = fast.detect(img,None)
    img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))

    # Print all default params
    print("Threshold: ", fast.getInt('threshold'))
    print("nonmaxSuppression: ", fast.getBool('nonmaxSuppression'))
    print("neighborhood: ", fast.getInt('type'))
    print("Total Keypoints with nonmaxSuppression: ", len(kp))

    #cv2.imwrite('fast_true.png',img2)
    plt.imshow(img2),plt.show()
    cv2.waitKey(1)

    # Disable nonmaxSuppression
    fast.setBool('nonmaxSuppression',0)
    kp = fast.detect(img,None)

    print("Total Keypoints without nonmaxSuppression: ", len(kp))

    img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))

    plt.imshow(img3),plt.show()

if exetasknum==2:
    # BRIEF (Binary Robust Independent Elementary Features)
    # One important point is that BRIEF is a feature descriptor, it doesn’t provide any method to find the features. 
    # So you will have to use any other feature detectors like SIFT, SURF etc. 
    # The paper recommends to use CenSurE which is a fast detector and BRIEF works even slightly better for CenSurE points than for SURF points.
    # In short, BRIEF is a faster method feature descriptor calculation and matching. 
    # It also provides high recognition rate unless there is large in-plane rotation.

    img = cv2.imread('simple.jpg',0)
    # Initiate STAR detector
    star = cv2.FeatureDetector_create("STAR")

    # Initiate BRIEF extractor
    brief = cv2.DescriptorExtractor_create("BRIEF")

    # find the keypoints with STAR
    kp = star.detect(img,None)

    # compute the descriptors with BRIEF
    kp, des = brief.compute(img, kp)

    print(brief.getInt('bytes'))
    print(des.shape)