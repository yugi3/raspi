import glob
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.axes_grid1 import ImageGrid
from natsort import natsorted


files = glob.glob("a Photos/*.jpg")
for i in natsorted(files):
    print("file:", i)
    print(i[9:])
   
"""
for n in range(len(files)):
    img = cv2.imread(files[n])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    plt.imshow(img)
"""