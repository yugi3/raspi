import cv2
import matplotlib.pyplot as plt
import torch
import urllib.request
import time
from natsort import natsorted
import glob
import csv


# デバイスを決める
device = torch.device("cuda")

# モデルのダウンロード
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

# 前処理用のトランスフォームをダウンロード
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform
process_time = []
files = glob.glob("b Photos/*.jpg")
for i in natsorted(files):
    allstart = time.perf_counter()

    # 画像を読み込む
    #filename = str(num) + '.jpg'
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 画像に前処理を施してバッチの準備
    input_batch = transform(img).to(device)

    # 推論の実行
    with torch.no_grad():
        prediction = midas(input_batch)
    
    allend = time.perf_counter()

    alltime = allend - allstart
    print("全体" + str(alltime) + "[sec]")
    process_time.append([alltime])

syori = 'depth2処理時間.csv'
with open(syori, "a", encoding="Shift_jis") as f: 
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(process_time)
    print(process_time)
    """
    # matplotlibで扱えるように推論結果を変換
    print(prediction.shape)
    output = prediction.squeeze()
    print(output.shape)
    output = output.cpu().numpy()
   
    # 入力画像と結果を表示
    fig, ax = plt.subplots(figsize=(output.shape[1]/10, output.shape[0]/10))
    ax.imshow(output, cmap='plasma')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    cname = 'depth ' + i[9:]
    plt.savefig(cname)
    """

