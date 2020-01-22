import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('C:\\Users\\sachi\\.vscode\\GitHubRepos\\OSCV_Exercises')
exetasknum = 3

if exetasknum==1:
    img = cv2.imread('messi5.jpg')
    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (50,50,450,290)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    plt.imshow(img),plt.colorbar(),plt.show()

if exetasknum==2:
    # I have not done it, but you can manually create mask

    # newmask is the mask image needs to be created
    newmask = cv2.imread('newmask.png',0)

    # whereever it is marked white (sure foreground), change mask=1
    # whereever it is marked black (sure background), change mask=0
    mask[newmask == 0] = 0
    mask[newmask == 255] = 1

    mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask[:,:,np.newaxis]
    plt.imshow(img),plt.colorbar(),plt.show()

if exetasknum==3:
    # this is from opencv interactive sample
    import sys

    class App():
        BLUE = [255,0,0]        # rectangle color
        RED = [0,0,255]         # PR BG
        GREEN = [0,255,0]       # PR FG
        BLACK = [0,0,0]         # sure BG
        WHITE = [255,255,255]   # sure FG

        DRAW_BG = {'color' : BLACK, 'val' : 0}
        DRAW_FG = {'color' : WHITE, 'val' : 1}
        DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
        DRAW_PR_BG = {'color' : RED, 'val' : 2}

        # setting up flags
        rect = (0,0,1,1)
        drawing = False         # flag for drawing curves
        rectangle = False       # flag for drawing rect
        rect_over = False       # flag to check if rect drawn
        rect_or_mask = 100      # flag for selecting rect or mask mode
        value = DRAW_FG         # drawing initialized to FG
        thickness = 3           # brush thickness

        def onmouse(self, event, x, y, flags, param):
            # Draw Rectangle
            if event == cv2.EVENT_RBUTTONDOWN:
                self.rectangle = True
                self.ix, self.iy = x,y

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.rectangle == True:
                    self.img = self.img2.copy()
                    cv2.rectangle(self.img, (self.ix, self.iy), (x, y), self.BLUE, 2)
                    self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
                    self.rect_or_mask = 0

            elif event == cv2.EVENT_RBUTTONUP:
                self.rectangle = False
                self.rect_over = True
                cv2.rectangle(self.img, (self.ix, self.iy), (x, y), self.BLUE, 2)
                self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
                self.rect_or_mask = 0
                print(" Now press the key 'n' a few times until no further change \n")

            # draw touchup curves

            if event == cv2.EVENT_LBUTTONDOWN:
                if self.rect_over == False:
                    print("first draw rectangle \n")
                else:
                    self.drawing = True
                    cv2.circle(self.img, (x,y), self.thickness, self.value['color'], -1)
                    cv2.circle(self.mask, (x,y), self.thickness, self.value['val'], -1)

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing == True:
                    cv2.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                    cv2.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)

            elif event == cv2.EVENT_LBUTTONUP:
                if self.drawing == True:
                    self.drawing = False
                    cv2.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                    cv2.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)

        def run(self):
            # Loading images
            if len(sys.argv) == 2:
                filename = sys.argv[1] # for drawing purposes
            else:
                print("No input image given, so loading default image, lena.jpg \n")
                print("Correct Usage: python grabcut.py <filename> \n")
                filename = 'messi5.jpg'

            self.img = cv2.imread(cv2.samples.findFile(filename))
            self.img2 = self.img.copy()                               # a copy of original image
            self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8) # mask initialized to PR_BG
            self.output = np.zeros(self.img.shape, np.uint8)           # output image to be shown

            # input and output windows
            cv2.namedWindow('output')
            cv2.namedWindow('input')
            cv2.setMouseCallback('input', self.onmouse)
            cv2.moveWindow('input', self.img.shape[1]+10,90)

            print(" Instructions: \n")
            print(" Draw a rectangle around the object using right mouse button \n")

            while(1):

                cv2.imshow('output', self.output)
                cv2.imshow('input', self.img)
                k = cv2.waitKey(1)

                # key bindings
                if k == 27:         # esc to exit
                    break
                elif k == ord('0'): # BG drawing
                    print(" mark background regions with left mouse button \n")
                    self.value = self.DRAW_BG
                elif k == ord('1'): # FG drawing
                    print(" mark foreground regions with left mouse button \n")
                    self.value = self.DRAW_FG
                elif k == ord('2'): # PR_BG drawing
                    self.value = self.DRAW_PR_BG
                elif k == ord('3'): # PR_FG drawing
                    self.value = self.DRAW_PR_FG
                elif k == ord('s'): # save image
                    bar = np.zeros((self.img.shape[0], 5, 3), np.uint8)
                    res = np.hstack((self.img2, bar, self.img, bar, self.output))
                    cv2.imwrite('grabcut_output.png', res)
                    print(" Result saved as image \n")
                elif k == ord('r'): # reset everything
                    print("resetting \n")
                    self.rect = (0,0,1,1)
                    self.drawing = False
                    self.rectangle = False
                    self.rect_or_mask = 100
                    self.rect_over = False
                    self.value = self.DRAW_FG
                    self.img = self.img2.copy()
                    self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8) # mask initialized to PR_BG
                    self.output = np.zeros(self.img.shape, np.uint8)           # output image to be shown
                elif k == ord('n'): # segment the image
                    print(""" For finer touchups, mark foreground and background after pressing keys 0-3
                    and again press 'n' \n""")
                    try:
                        if (self.rect_or_mask == 0):         # grabcut with rect
                            bgdmodel = np.zeros((1, 65), np.float64)
                            fgdmodel = np.zeros((1, 65), np.float64)
                            cv2.grabCut(self.img2, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
                            self.rect_or_mask = 1
                        elif self.rect_or_mask == 1:         # grabcut with mask
                            bgdmodel = np.zeros((1, 65), np.float64)
                            fgdmodel = np.zeros((1, 65), np.float64)
                            cv2.grabCut(self.img2, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
                    except:
                        import traceback
                        traceback.print_exc()

                mask2 = np.where((self.mask==1) + (self.mask==3), 255, 0).astype('uint8')
                self.output = cv2.bitwise_and(self.img2, self.img2, mask=mask2)

            print('Done')


    if __name__ == '__main__':
        print(__doc__)
        App().run()
        cv2.destroyAllWindows()