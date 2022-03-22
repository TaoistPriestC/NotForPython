import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
img1 = Image.open('Pic1.png').resize((350,350),Image.BICUBIC).convert("RGB")
img2 = Image.open('Pic4.png').resize((350,350),Image.BICUBIC).convert("RGB")
# print(img1.size)
# print(img2.size)

new_img = Image.blend(img1, img2, 0.03)


TITLE_FONT_SIZE = 12

config = {
            "font.family": 'serif',
            "font.size": 12,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
}
plt.rcParams.update(config)

plt.subplot(1,3,1)
plt.imshow(img1)
plt.title("虎 猫", fontsize = TITLE_FONT_SIZE)
plt.subplot(1,3,2)
plt.imshow(img2)
plt.title("噪 声", fontsize = TITLE_FONT_SIZE)
plt.subplot(1,3,3)
plt.title("键 盘", fontsize = TITLE_FONT_SIZE)
plt.imshow(new_img)
plt.show()