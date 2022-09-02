
import cv2
import numpy as np
img=cv2.imread('screen1.jpg',cv2.IMREAD_COLOR)
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
template=cv2.imread('wechat_logo.png',cv2.IMREAD_GRAYSCALE)
height,width=template.shape[:2]
#print(height,width)
res=cv2.matchTemplate(imgGray,template,cv2.TM_CCOEFF_NORMED)
thresthold=0.1
loc = np.where(res <= thresthold)
for pt in zip(*loc[::-1]):
    right_bottom = (pt[0] + width, pt[1] + height)
    cv2.rectangle(img, pt, right_bottom, (0, 0, 255), 2)
cv2.imshow('src',img)
cv2.imshow('template',template)
cv2.imshow('result',res)
cv2.waitKey(0)
