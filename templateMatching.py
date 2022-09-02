import cv2 as cv
import numpy as np
from timeit import default_timer as timer
def template_demo():
    tpl = cv.imread("1.jpg")
    target = cv.imread("xz.jpg")
    cv.imshow("template image",tpl)
    cv.imshow("target image",target)
    methods = [cv.TM_SQDIFF_NORMED,cv.TM_CCORR_NORMED,cv.TM_CCOEFF_NORMED]#各种匹配算法
    th,tw = tpl.shape[:2]#获取模板图像的高宽
    thresthold1=0.15
    thresthold2=0.99
    for md in methods:
        result = cv.matchTemplate(target,tpl,md)
        # result是我们各种算法下匹配后的图像
        # cv.imshow("%s"%md,result)
        #获取的是每种公式中计算出来的值，每个像素点都对应一个值
        min_val,max_val,min_loc,max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = np.where(result <= thresthold1)   #tl是左上角点
        else:
            tl = np.where(result >= thresthold2)
        for pt in zip(*tl[::-1]):
            right_bottom = (pt[0] + tw, pt[1] + th)
            cv.rectangle(target, pt, right_bottom, (0, 0, 255), 2)
        # br = (tl[0]+tw,tl[1]+th)    #右下点
        # cv.rectangle(target,tl,br,(0,0,255),2)#画矩形
        cv.imshow("match-%s"%md,target)
        toc = timer()
        print(toc - tic)

tic=timer()
src = cv.imread("castal.jpg")  #读取图片
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE) #创建GUI窗口,形式为自适应
cv.imshow("input image",src)#通过名字将图像和窗口联系
template_demo()
cv.waitKey(0) #等待用户操作，里面等待参数是毫秒，我们填写0，代表是永远，等待用户操作
cv.destroyAllWindows() #销毁所有窗口