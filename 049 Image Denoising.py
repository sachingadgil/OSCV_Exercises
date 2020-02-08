import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 3

if exetasknum==1:
    # cv2.fastNlMeansDenoisingColored()
    img = cv2.imread('test.jpg')

    dst = cv2.fastNlMeansDenoisingColored(img,None,5,5,5,10)
    # above parameters worked for my image
    # original settings were dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    plt.subplot(121),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(122),plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.show()

if exetasknum==2:
    # cv2.fastNlMeansDenoisingMulti()
    # This did not reduced noise for me
    cap = cv2.VideoCapture('output.avi')

    # create a list of first 5 frames
    img = [cap.read()[1] for i in range(5)]

    # convert all to grayscale
    gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in img]

    # convert all to float64
    gray = [np.float64(i) for i in gray]

    # create a noise of variance 25
    noise = np.random.randn(*gray[1].shape)*10

    # Add this noise to images
    noisy = [i+noise for i in gray]

    # Convert back to uint8
    noisy = [np.uint8(np.clip(i,0,255)) for i in noisy]

    # Denoise 3rd frame considering all the 5 frames
    dst = cv2.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 4, 7, 35)

    plt.subplot(131),plt.imshow(gray[2],'gray')
    plt.subplot(132),plt.imshow(noisy[2],'gray')
    plt.subplot(133),plt.imshow(dst,'gray')
    plt.show()

if exetasknum==3:
    # Trying colored blurring in camera feed
    # it works but heavy on cpu
    cap = cv2.VideoCapture(0)
    
    def make_720p():
        cap.set(3, 1280)
        cap.set(4, 720)
    make_720p()

    while(1):
        ret, frame = cap.read()

        dst = cv2.fastNlMeansDenoisingColored(frame,None,10,10,3,12)

        cv2.imshow('original', frame)
        cv2.imshow('DenoisingColored',dst)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()