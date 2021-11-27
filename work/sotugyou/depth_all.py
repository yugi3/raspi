import cv2
import matplotlib.pyplot as plt
import torch
import urllib.request
import time
from natsort import natsorted
import glob


# デバイスを決める
device = torch.device("cpu")

# モデルのダウンロード
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

# 前処理用のトランスフォームをダウンロード
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

files = glob.glob("1 Photos/*.jpg")
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
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax1.axis('off')
    ax2.imshow(output, cmap='gray')
    ax2.axis('off')
    cname = '/home/ubuntu/work/sotugyou/photo/'+'hi ' + i[9:]
    plt.savefig(cname)

    fig, ax = plt.subplots(figsize=(output.shape[1]/10, output.shape[0]/10))
    ax.imshow(output,cmap='gray')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    cname = '/home/ubuntu/work/sotugyou/photo/'+'depth ' + i[9:]
    plt.savefig(cname)
    
    
    

