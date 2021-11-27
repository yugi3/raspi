import cv2
import matplotlib.pyplot as plt
import torch
import urllib.request
import time


# デバイスを決める
device = torch.device("cuda")

# モデルのダウンロード
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

# 前処理用のトランスフォームをダウンロード
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform


allstart = time.perf_counter()
# 画像を読み込む
filename = "10.jpg"
img = cv2.imread(filename)
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
fig, ax = plt.subplots()
ax.imshow(output, cmap='plasma')
cname = 'depth ' 
plt.savefig(cname)

