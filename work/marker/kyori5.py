# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import time
import csv


# 抽出したいカラーボールの色指定
# 0 <= h <= 179 (色相)　OpenCVではmax=179なのでR:0(180),G:60,B:120となる
# 0 <= s <= 255 (彩度)　黒や白の値が抽出されるときはこの閾値を大きくする
# 0 <= v <= 255 (明度)　これが大きいと明るく，小さいと暗い
c_min = np.array([97, 50, 0])
c_max = np.array([117, 255, 255])    
time_list = []
# 膨張化用のカーネル
k = np.ones((5,5),np.uint8)

def color_track(im,h_min,h_max):
    global time_list
    time_list.append(['RGB色空間からHSV色空間に変換', time.perf_counter()])
    im_h = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)  # RGB色空間からHSV色空間に変換
    time_list.append(['RGB色空間からHSV色空間に変換', time.perf_counter()])
    time_list.append(['マスク画像の生成', time.perf_counter()])
    mask = cv2.inRange(im_h,h_min,h_max,)       # マスク画像の生成
    time_list.append(['マスク画像の生成', time.perf_counter()])
    time_list.append(['平滑化', time.perf_counter()])
    mask = cv2.medianBlur(mask,7)               # 平滑化
    time_list.append(['平滑化', time.perf_counter()])
    time_list.append(['膨張化', time.perf_counter()])
    mask = cv2.dilate(mask,k,iterations=2)      # 膨張化
    time_list.append(['膨張化', time.perf_counter()])
    time_list.append(['色領域抽出', time.perf_counter()])
    im_c = cv2.bitwise_and(im,im,mask=mask)     # 色領域抽出
    time_list.append(['色領域抽出', time.perf_counter()])
    return mask#,im_c
def    index_emax(cnt):
    max_num = 0
    max_i = -1
    for i in range(len(cnt)):
        cnt_num=len(cnt[i])
        if cnt_num > max_num:
            max_num = cnt_num
            max_i = i
    return max_i
def    main():
    global time_list
    allstart = time.perf_counter()
    L1 = 200  # カメラと測定場所の距離　200[mm]
    h1 = 798  # 測定場所での画面サイズ 220 [ピクセル]
    H =  50   # ボールの直径[mm]
    Z =  200  # 200mm先のz位置[mm
    # 入力画像の取得
    name ='20.jpg'
    time_list.append(['画像取得', time.perf_counter()])
    im =   cv2.imread(name)      # カラーボールのトラッキング
    time_list.append(['画像取得', time.perf_counter()])
    time_list.append(['リサイズ', time.perf_counter()])
    height = im.shape[0]
    width = im.shape[1]
    # 画像のサイズを変更
    # 第一引数：サイズを変更する画像
    # 第二引数：変更後の幅
    # 第三引数：変更後の高さ
    size = 50 #サイズ指定
    im= cv2.resize(im,(width//size, height//size))
    time_list.append(['リサイズ', time.perf_counter()])
    mask1 = color_track(im,c_min,c_max)
    mask2 = color_track(im,c_min,c_max)
    # 色領域の輪郭を抽出
    time_list.append(['色領域の輪郭を抽出', time.perf_counter()])
    cnt, hierarchy = cv2.findContours(mask2,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    time_list.append(['色領域の輪郭を抽出', time.perf_counter()])
    #cnt = cv2.findContours(mask2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
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
    # 色領域の高さ計算
    time_list.append(['距離計算', time.perf_counter()])
    mask = cv2.bitwise_and(mask1,mask1,mask=mask2)
    mask = cv2.Canny(mask2,100,200)
    y,x = np.where(mask == 255)
    if len(y)!= 0:
        # 対象物体の画像上の高さh2を計算
        ymax,ymin = np.amax(y),np.amin(y)
        xmax,xmin = np.amax(x),np.amin(x)
        #高さと幅の大きいほうを直径として使う
        h2 = max([(ymax - ymin),(xmax - xmin)])
        #中心位置を直径をもとに補正（要修正）
        cx = cx + int((h2 - (xmax - xmin))/2)
        cy = cy + int((h2 - (ymax - ymin))/2)
        if float(h2) !=0:
            # 奥行きL2を計算
            L2 = (h1/float(h2))*L1
            # 1px当たりの大きさを計算
            a = H/float(h2)
            # 三次元位置（X, Y, Z）を計算
            X = (cx-320)*a
            Y = (240-cy)*a
            if L2 > X:
                Z = math.sqrt(L2*L2-X*X)
            X,Y,Z,L2 = round(X),round(Y),round(Z),round(L2)
        time_list.append(['距離計算', time.perf_counter()])
        """
        alltime_list = time.perf_counter()
        # 処理時間算出
        alltime = alltime_list- allstart
        print("全体" + str(alltime) + "[sec]")
        print(time_list)
        process_time = []
        process_time.append([name])
        process_time.append(['全体',alltime])
        i = 0
        while i < len(time_list):
            pname = time_list[i][0]
            ans = time_list[i+1][1] - time_list[i][1]
            process_time.append([pname, ans])
            print(pname,ans)
            i= i+2
        print(process_time)
       
        #処理csvファイル作成
        with open("処理時間.csv", "a", encoding="Shift_jis") as f: 
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(process_time)
        #推定csvファイル作成
        long = name.replace('.jpg','')
        est = str(L2)   #推定距離
        est=  est[:2] + '.' + est[2:]
        #ファイルを読み取り
        with open("推定結果.csv","a") as f:
            with open("推定結果.csv","r+") as f: 
                reader = csv.reader(f)
                l = [row for row in reader]
                print(l)
        #ファイルが空の場合
        if len(l) == 0:         
            l.append([long])
            l.append([est])
            with open("推定結果.csv","w") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerows(l)
        #ファイルが書き込まれていた場合
        else:
            with open("推定結果.csv","w") as f:
                l[0].append(long)
                l[1].append(est)
                writer = csv.writer(f, lineterminator="\n")
                writer.writerows(l)
        """
        # 結果表示
        cv2.circle(im,(cx,cy),5, (0,0,255), -1)
        cv2.circle(im,(cx,cy),int(h2/2), (255,0,255), 1)
        cv2.putText(im,"X: "+str(X)+"[mm]",(30,20),1,1.5,(70,70,220),2)
        cv2.putText(im,"Y: "+str(Y)+"[mm]",(30,50),1,1.5,(70,70,220),2)
        cv2.putText(im,"Z: "+str(Z)+"[mm]",(30,80),1,1.5,(70,70,220),2)
        cv2.putText(im,"h2: "+str(h2)+"[pixcel]",(30,120),1,1.5,(220,70,90),2)
        cv2.putText(im,"L2: "+str(L2)+"[mm]",(30,160),1,1.5,(220,70,90),2)
        print("h2: "+str(h2)+"[pixcel]")
        print("L2: "+str(L2)+"[mm]")
        
        """"
        height = im.shape[0]
        width = im.shape[1]
        # 画像のサイズを変更
        # 第一引数：サイズを変更する画像
        # 第二引数：変更後の幅
        # 第三引数：変更後の高さ
        im= cv2.resize(im,(width//4, height//4))
        mask2 = cv2.resize(mask2,(width//4, height//4))
        """
        cv2.namedWindow("Camera", 1) 
        cv2.namedWindow("Mask", 1) 
        cv2.imshow("Camera",im)
        cv2.imshow("Mask",mask2)
        cname = str(size) + 'c' + name
        mname = str(size) + 'm' + name
        print(cname)
        cv2.imwrite(cname,im)
        cv2.imwrite(mname,mask2)
    
if __name__ == '__main__':
    main()
