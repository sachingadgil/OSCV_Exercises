import cv2
import numpy as np
import random
from PIL import Image
import tkinter as tk
from tkinter import simpledialog
from datetime import datetime
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum=1
bgcolor='Transparent'

def imgnamedialogue():
    application_window = tk.Tk()
    application_window.lift()
    answer = simpledialog.askstring("Input", "What is your first name?", parent=application_window)
    return answer

if exetasknum==1:
    # mouse callback function
    #original function
    def draw_circle(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img,(x,y),100,(255,100,255),-1)
    #myfunction with randomization
    def draw_randcircle(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONUP:
            cv2.circle(img,(x,y), random.randint(30,50),(random.randint(0,255),random.randint(0,255),random.randint(0,255)),-1)
    # Create a black image, a window and bind the function to window
    img = np.zeros((900,1600,3), np.uint8)
    img.fill(255) # to make the image backgroud white
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_randcircle)

    while(1):
        cv2.imshow('image',img)
        if cv2.waitKey(20) & 0xFF == 27:
            #filename=input("enter image file name")
            filename = imgnamedialogue()
            #stamp=str(datetime.now())
            filenamewithpath=os.getcwd()+"\\"+filename+".png"
            print(filenamewithpath)
            cv2.imwrite(filenamewithpath,img)
            break
    cv2.destroyAllWindows()

if bgcolor=='Transparent':
    img2 = Image.open(filenamewithpath)
    img2 = img2.convert("RGBA")
    pixdata = img2.load()
    width, height = img2.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y] == (255, 255, 255, 255):
                pixdata[x, y] = (255, 255, 255, 0)
    img2.save(filenamewithpath, "PNG")