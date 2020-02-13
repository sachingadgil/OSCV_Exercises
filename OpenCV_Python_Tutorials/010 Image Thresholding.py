import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 4

if exetasknum==1:
    #Simple Thresholding
    img = cv2.imread('Webcam.jpg',0)
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()

if exetasknum==2:
    #Adaptive Thresholding
    img = cv2.imread('Webcam.jpg',0)
    img = cv2.medianBlur(img,5)

    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)

    titles = ['Original Image', 'Global Thresholding (v = 127)',
                'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]

    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

if exetasknum==3:
    #Trying image from webcam with stabilization - stabilization did not work
    cap = cv2.VideoCapture(0)

    def make_720p():
        cap.set(3, 1280)
        cap.set(4, 720)
    make_720p()
    
    idx = 0
    bufflen = 8

    while(1):
        pos = idx % bufflen
        _, frame_i = cap.read()
        frame_s = cv2.cvtColor(frame_i, cv2.COLOR_BGR2GRAY)
        frame_s = cv2.medianBlur(frame_s,5)
        frame = cv2.adaptiveThreshold(frame_s,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,2)

        if (idx>=bufflen):
            if pos==0:
                frame_b0 = frame.copy()
                frame_b0 = np.int32(frame_b0) # this conversion is required to get the summation / average right and not bound by uint8 size
            elif pos==1:
                frame_b1 = frame.copy()
                frame_b1 = np.int32(frame_b1)
            elif pos==2:
                frame_b2 = frame.copy()
                frame_b2 = np.int32(frame_b2)
            elif pos==3:
                frame_b3 = frame.copy()
                frame_b3 = np.int32(frame_b3)
            elif pos==4:
                frame_b4 = frame.copy()
                frame_b4 = np.int32(frame_b4)
            elif pos==5:
                frame_b5 = frame.copy()
                frame_b5 = np.int32(frame_b5)
            elif pos==6:
                frame_b6 = frame.copy()
                frame_b6 = np.int32(frame_b6)
            elif pos==7:
                frame_b7 = frame.copy()
                frame_b7 = np.int32(frame_b7)

            frame_t = (np.array(frame_b0) + np.array(frame_b1) + np.array(frame_b2) + np.array(frame_b3) + np.array(frame_b4) + np.array(frame_b5) + np.array(frame_b6) + np.array(frame_b7))/8
            frame_f = np.uint8(frame_t)
        else:
            frame_b0 = frame_b1 = frame_b2 = frame_b3 = frame.copy()
            frame_b4 = frame_b5 = frame_b6 = frame_b7 = frame.copy()
            frame_t = frame_r = frame_f = frame.copy()

        cv2.imshow('Gaussian',frame_f)
        cv2.imshow('not_Stabilized',frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()    

if exetasknum==4:
    #Trying image from webcam 
    cap = cv2.VideoCapture(0)

    def make_720p():
        cap.set(3, 1280)
        cap.set(4, 720)
    make_720p()

    idx = 0
    bufflen = 4

    while(1):
        _, frame_i = cap.read()
        frame_s = cv2.cvtColor(frame_i, cv2.COLOR_BGR2GRAY)
        frame_s = cv2.medianBlur(frame_s,3)
        frame = cv2.adaptiveThreshold(frame_s,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,15,3)
        if (idx>=bufflen):
            #frame_b[pos] = frame.copy() # wanted to use a loop like this, but used hardcoding instead for lack of time
            if pos==0:
                 frame_b0 = frame.copy()
                 frame_b0 = np.int32(frame_b0) # this conversion is required to get the summation / average right and not bound by uint8 size
            elif pos==1:
                 frame_b1 = frame.copy()
                 frame_b1 = np.int32(frame_b1)
            elif pos==2:
                 frame_b2 = frame.copy()
                 frame_b2 = np.int32(frame_b2)
            elif pos==3:
                 frame_b3 = frame.copy()
                 frame_b3 = np.int32(frame_b3)

            frame_t = (np.array(frame_b0) + np.array(frame_b1) + np.array(frame_b2) + np.array(frame_b3))/4
            frame_f = np.uint8(frame_t)

            #print("frame", frame_r.shape)
            #print("frame_b0", frame_b0.shape)
            #print("frame_b1", frame_b1.shape)
            #print("frame_b2", frame_b2.shape)
            #print("frame_t", frame_t.shape)
            #print("frame_r", frame_r.shape)
        else:
            frame_b0 = frame_b1 = frame_b2 = frame_b3 = frame.copy()
            frame_t = frame_r = frame_f = frame.copy()
        
        frame_a = cv2.medianBlur(frame,3)
        frame_g = cv2.adaptiveThreshold(frame_a,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,15,3)

        cv2.imshow('not_Stabilized',frame_g)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()    

if exetasknum==5:
    #example provided to explain Otsu’s Binarization
    img = cv2.imread('otsu.jpg',0)

    # global thresholding
    ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    # Otsu's thresholding
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # plot all the images and their histograms
    images = [img, 0, th1,
            img, 0, th2,
            blur, 0, th3]
    titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
            'Original Noisy Image','Histogram',"Otsu's Thresholding",
            'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

    for i in range(3):
        plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.show()

if exetasknum==6:
    #python implementation of Otsu’s Binarization
    img = cv2.imread('otsu.jpg',0)
    blur = cv2.GaussianBlur(img,(5,5),0)

    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([blur],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.max()
    Q = hist_norm.cumsum()

    bins = np.arange(256)

    fn_min = np.inf
    thresh = -1

    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        b1,b2 = np.hsplit(bins,[i]) # weights

        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i

    # find otsu's threshold value with OpenCV function
    ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print (thresh, ret)

if exetasknum==7:
    #adopting otsu in camera image - but only splits the image in black and white
    cap = cv2.VideoCapture(0)

    def make_720p():
        cap.set(3, 1280)
        cap.set(4, 720)
    make_720p()
    while(1):
        _, frame_i = cap.read()
        frame_s = cv2.cvtColor(frame_i, cv2.COLOR_BGR2GRAY)
        frame_s = cv2.GaussianBlur(frame_s,(5,5),0)
        ret, frame = cv2.threshold(frame_s,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow('Stabilized',frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()