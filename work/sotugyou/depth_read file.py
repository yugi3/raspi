import cv2
import matplotlib.pyplot as plt
import torch
import urllib.request
import time
from natsort import natsorted
import glob


# デバイスを決める
device = torch.device("cuda")

# モデルのダウンロード
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

# 前処理用のトランスフォームをダウンロード
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

files = glob.glob("b Photos/*.jpg")
for i in natsorted(files):
    allstart = time.perf_counter()

    # 画像を読み込む
    #filename = str(num) + '.jpg'
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 画像に前処理を施してバッチの準備
    input_batch = transform(img).to(device)
    print(input_batch.shape)

    # 推論の実行
    with torch.no_grad():
        prediction = midas(input_batch)
    
    allend = time.perf_counter()

    # matplotlibで扱えるように推論結果を変換
    print(prediction.shape)
    output = prediction.squeeze()
    print(output.shape)

    output = output.cpu().numpy()

    # 入力画像と結果を表示
    alltime = allend - allstart
    print("全体" + str(alltime) + "[sec]")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax1.axis('off')
    ax2.imshow(output, cmap='plasma')
    ax2.axis('off')
    cname = 'c' + i[9:]
    plt.savefig(cname)

