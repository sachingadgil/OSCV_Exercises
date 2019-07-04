import numpy as np
import cv2
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 1

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


while(1):

    # Take each frame
    _, frame = cap.read()

    # Use this if frame size of captured video after capturing needs to be resized
    #frame_s = rescale_frame(frame, percent=30)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of desired color range in HSV
    # Different applications use different scales for HSV. For example gimp uses H = 0-360, S = 0-100 and V = 0-100.
    # OpenCV uses  H: 0-179, S: 0-255, V: 0-255 (H is basically half the value of angle appearing on HSV colorwheel)
    lower_color = np.array([130,50,50])
    upper_color = np.array([170,255,255])

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