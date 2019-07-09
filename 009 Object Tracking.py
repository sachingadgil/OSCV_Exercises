import numpy as np
import cv2
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 2

# If you need to print color conversion operation options
#colorspaceflages = [i for i in dir(cv2) if i.startswith('COLOR_')]
#print(colorspaceflages)


cap = cv2.VideoCapture(0)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)
make_720p()

# General function to change resolution of capture video
def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

# Function to rescale the video captured
def rescale_frame(frame, percent=75):
    scale_percent = 75
    width = int(frame.shape[1] * scale_percent / 100)    
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

if exetasknum==1:
    while(1):

        # Take each frame
        _, frame = cap.read()
        #print(frame.shape()) # Does not work - tuple not callable
        # Use this if frame size of captured video after capturing needs to be resized
        #frame_s = rescale_frame(frame, percent=30)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of desired color range in HSV
        # Different applications use different scales for HSV. For example gimp uses H = 0-360, S = 0-100 and V = 0-100.
        # OpenCV uses  H: 0-179, S: 0-255, V: 0-255 (H is basically half the value of angle appearing on HSV colorwheel)
        lower_color = np.array([45,25,25])
        upper_color = np.array([75,255,255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)

        cv2.imshow('frame',frame)
        #cv2.imshow('frame_s',frame_s)    
        #cv2.imshow('mask',mask)
        cv2.imshow('res',res)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

if exetasknum==2:
    idx = 0 # not worrying about int or long as pythong's ints can go to to a large count
    bufflen = 4 # more than 3 is not possible if addWeighted function is used
    while(1):
        # trying to stabilize flicker by averaging 5 consicutavie frames inputs
        pos = idx % bufflen

        # Take each frame
        _, frame = cap.read()
        #frame = np.uint8(frame)  # this line has no effect as the captured image is in uint8 pixel value format

        # create frame buffer
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
            #frame_t = (frame_b0 + frame_b1 + frame_b2 + frame_b3)/4  # Alternate way to do the same thing
            #frame_r = np.rint(frame_t)   # Alternate way to do the same thing
            #frame_r = frame_r.astype(int)   # Alternate way to do the same thing
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
            #print("frame_r in else", frame_r.shape)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame_f, cv2.COLOR_BGR2HSV)

        # define range of desired color range in HSV
        # Different applications use different scales for HSV. For example gimp uses H = 0-360, S = 0-100 and V = 0-100.
        # OpenCV uses  H: 0-179, S: 0-255, V: 0-255 (H is basically half the value of angle appearing on HSV colorwheel)
        #lower_color = np.array([1,30,5])   # for skin
        #upper_color = np.array([13,255,255]) # for skin
        lower_color = np.array([40,30,5])
        upper_color = np.array([80,255,255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame_f,frame_f, mask= mask)

        #cv2.imshow('frame_f',frame_f)
        cv2.imshow('frame',frame)    
        #cv2.imshow('mask',mask)
        cv2.imshow('res',res)

        idx=idx+1

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
