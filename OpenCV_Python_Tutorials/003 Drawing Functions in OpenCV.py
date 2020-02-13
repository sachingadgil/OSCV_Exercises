#Drawing functions
import numpy as np
import cv2

# Create a black image
img = np.zeros((512,512,3), np.uint8)

# Draw a diagonal blue line with thickness of 5 px
img = cv2.line(img,(0,0),(511,511),(0,150,255),5)

img = cv2.rectangle(img,(384,0),(510,127),(0,255,0),-1)

img = cv2.circle(img,(447,63), 64, (0,0,255), -1)

img = cv2.ellipse(img,(256,256),(100,50),0,0,270,200,5)

pts = np.array([[110,15],[120,130],[70,320],[150,110]], np.int32)
pts = pts.reshape((-1,1,2))
img = cv2.polylines(img,[pts],True,(0,255,255))

font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img,'OpenCV',(10,450), font, 3,(255,255,255),3,cv2.LINE_AA)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#print(pts)