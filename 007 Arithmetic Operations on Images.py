import numpy as np
import cv2
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 4

img1 = cv2.imread('B2DBy.jpg')
img2 = cv2.imread('opencv-logo-white.png')

if exetasknum==1:
    #Image Addition
    x = np.uint8([250])
    y = np.uint8([10])
    print(cv2.add(x,y)) # 250+10 = 260 => 255
    print(x+y)          # 250+10 = 260 % 256 = 4

if exetasknum==2:
    #Image Blending
    print(img1.shape)
    print(img2.shape)
    
    dst = cv2.addWeighted(img1,0.7,img2,0.3,0) # This will not work unless image sizes are same

    cv2.imshow('dst',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if exetasknum==3:
    #Bitwise Operations - Add logo on image

    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols ]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    print(ret)
    print(mask.shape)
    print(mask)
    mask_inv = cv2.bitwise_not(mask)
    print(mask_inv.shape)
    print(mask_inv)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    img1[0:rows, 0:cols ] = dst

    cv2.imshow('res',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if exetasknum==4:
    #Bitwise Operations - Add blended / watermarked logo

    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols ]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    # Take only region of logo from reference image.
    img1_rg = cv2.bitwise_and(roi,roi,mask = mask)

    # Create blended copy of logo
    img2_mg = cv2.addWeighted(img1_rg,0.7,img2_fg,0.3,0)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_mg)
    img1[0:rows, 0:cols ] = dst

    cv2.imshow('res',img1)
    #cv2.imshow('res',img2_mg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()