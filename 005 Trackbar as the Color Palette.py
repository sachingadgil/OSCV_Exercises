import numpy as np
import cv2
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum=2
drawing = False

if exetasknum==1:
    def nothing(x):
        pass

    # Create a black image, a window
    img = np.zeros((800,600,3), np.uint8)
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('R','image',0,255,nothing)
    cv2.createTrackbar('G','image',0,255,nothing)
    cv2.createTrackbar('B','image',0,255,nothing)

    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image',0,1,nothing)

    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R','image')
        g = cv2.getTrackbarPos('G','image')
        b = cv2.getTrackbarPos('B','image')
        s = cv2.getTrackbarPos(switch,'image')

        if s == 0:
            img[:] = 0
        else:
            img[:] = [b,g,r]

    cv2.destroyAllWindows()

if exetasknum==2:
    def nothing(x):
        pass

    # Create a black image, a window
    img = np.zeros((768,1024,3), np.uint8)
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('R','image',0,255,nothing)
    cv2.createTrackbar('G','image',0,255,nothing)
    cv2.createTrackbar('B','image',0,255,nothing)
    cv2.createTrackbar('W','image',0,50,nothing)
    
    def draw_brush(event,x,y,flags,param):
        global drawing, b, g, r, w
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            cv2.circle(img,(x,y),w,(b,g,r),-1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                p, q = x, y
                if p!=x & q!=y:
                    cv2.circle(img,(x,y),w,(b,g,r),-1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
    # Create a window and bind the function to window
    cv2.setMouseCallback('image',draw_brush)

    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R','image')
        g = cv2.getTrackbarPos('G','image')
        b = cv2.getTrackbarPos('B','image')
        w = cv2.getTrackbarPos('W','image')

    cv2.destroyAllWindows()    