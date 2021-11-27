# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math


# ???o?μ???￠¶?I???F?w?e
# 0 <= h <= 179 (?F??) OpenCV??Imax=179???AR:0(180),G:60,B:120???e
# 0 <= s <= 255 (??x) ??????l?a???o?3?e?e??≪??±?I?l?d???-?・?e
# 0 <= v <= 255 (???x) ?±?????￠?????,?￢?3?￠??A￠
c_min = np.array([97, 50, 50])
c_max = np.array([117, 255, 255])    

# ?c?￡?≫?p??°EU
k = np.ones((5,5),np.uint8)

def color_track(im,h_min,h_max):
    im_h = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)  # RGB?F?o??c?cHSV?F?o????・
    mask = cv2.inRange(im_h,h_min,h_max,)       # ?，????¶?￢
    mask = cv2.medianBlur(mask,7)               # ?????≫
    #mask = cv2.dilate(mask,k,iterations=2)      # ?c?￡?≫
    #im_c = cv2.bitwise_and(im,im,mask=mask)     # ?F????o
    return mask#,im_c

def index_emax(cnt):
    max_num = 0
    max_i = -1
    for i in range(len(cnt)):
        cnt_num=len(cnt[i])
        if cnt_num > max_num:
            max_num = cnt_num
            max_i = i

    return max_i

def main():

    L1 = 200  # ¶O??a?e??????￡ 200[mm]
    h1 = 220  # ?a?e?????a??2?T 220 [E??U]
    H =  55   # I?????a[mm]
    Z =  200  # 200mm?a?Iz??u[mm]
    #cap = cv2.VideoCapture(0)
    #if cap.isOpened() is False:
    #   raise("IO Error")

    while True:
        # ?u?????擾
        im =  cv2.imread('k20r.jpg')

        # ¶?I??IA?・?T
        mask1 = color_track(im,c_min,c_max)
        mask2 = color_track(im,c_min,c_max)

        # ?F??a???s? o
        cnt, hierarchy = cv2.findContours(mask2,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = cv2.findContours(mask2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
        n = index_emax(cnt)

        if n != -1:
            hull = cv2.convexHull(cnt[n])
            mask2[:] = 0
            cv2.drawContours(mask2,[hull],0,(255),-1)
            cv2.drawContours(im,[hull],0,(0,200,0),2)
            cv2.drawContours(mask2,[hull],0,(0,200,0),2)
            M1 = cv2.moments(cnt[n])
            if M1["m00"] != 0:
               cx,cy = int(M1["m10"]/M1["m00"]),int(M1["m01"]/M1["m00"])
            else:
                cx,cy = 320,240
        else:
            cx,cy = 320,240

        # ?F??a????3?v?Z
        mask = cv2.bitwise_and(mask1,mask1,mask=mask2)
        mask = cv2.Canny(mask2,100,200)
        y,x = np.where(mask == 255)
        if len(y)!= 0:
            # ???¨?????a????3h2?d?v?Z
            ymax,ymin = np.amax(y),np.amin(y)
            xmax,xmin = np.amax(x),np.amin(x)
            
            #???3???????￠???? a??μ??g??
            h2 = max([(ymax - ymin),(xmax - xmin)])

            #???S??u? a?d?a????(?v?C?3)
            cx = cx + int((h2 - (xmax - xmin))/2)
            cy = cy + int((h2 - (ymax - ymin))/2)
            
            if float(h2) !=0:
                # ???s?≪L2?d?v?Z
                L2 = (h1/float(h2))*L1
                # 1px?????e????3?d?v?Z
                a = H/float(h2)
                # ?O???3??u(X, Y, Z)?d?v?Z
                X = (cx-320)*a
                Y = (240-cy)*a
                if L2 > X:
                    Z = math.sqrt(L2*L2-X*X)
                X,Y,Z,L2 = round(X),round(Y),round(Z),round(L2)
            # ????\?|
            cv2.circle(im,(cx,cy),5, (0,0,255), -1)
            cv2.circle(im,(cx,cy),int(h2/2), (255,0,255), 1)
            cv2.putText(im,"X: "+str(X)+"[mm]",(30,20),1,1.5,(70,70,220),2)
            cv2.putText(im,"Y: "+str(Y)+"[mm]",(30,50),1,1.5,(70,70,220),2)
            cv2.putText(im,"Z: "+str(Z)+"[mm]",(30,80),1,1.5,(70,70,220),2)
            cv2.putText(im,"h2: "+str(h2)+"[pixcel]",(30,120),1,1.5,(220,70,90),2)
            cv2.putText(im,"L2: "+str(L2)+"[mm]",(30,160),1,1.5,(220,70,90),2)
            cv2.imshow("Camera",im)
            cv2.imshow("Mask",mask2)
        # ・°?a???3???c?I?c?甲?￣?e
        if cv2.waitKey(10) > 0:
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()
