import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.metrics import structural_similarity as ssim
 

img1 = Image.open('Pic1.png')
img2 = Image.open('Pic2.png')
img3 = Image.open('Pic3.png')

img1 = img1.resize((350,350),Image.BICUBIC)
img2 = img2.resize((350,350),Image.BICUBIC)
img3 = img3.resize((350,350),Image.BICUBIC)

# img1.save("Pic1.png")
# img2.save("Pic2.png")
# img3.save("Pic3.png")

print(img1.size)
print(img2.size)
print(img3.size)


def showImage():
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.subplot(131)
    plt.imshow(img1)
    plt.subplot(132)
    plt.imshow(img2)
    plt.subplot(133)
    plt.imshow(img3)
    plt.show()


def getDist():
    _img1 = np.array(img1)
    _img2 = np.array(img2)
    _img3 = np.array(img3)
    # dist_12 = np.linalg.norm((_img1 - _img2)/255, ord = 1)
    # dist_13 = np.linalg.norm((_img1 - _img3)/255, ord = 1)
    dist_12 = ssim(_img1, _img2, multichannel=True)
    dist_13 = ssim(_img1, _img3, multichannel=True)
    print("dist_12 = ", dist_12)
    print("dist_13 = ", dist_13)


if __name__ == '__main__':
    # 这份代码主要是用来测试两张图片的相似程度, 最初使用了欧式距离,实验结果
    # 发现使用欧式距离测出虎猫与埃及猫的距离, 竟然大于虎猫与键盘的距离, 对于自行构造的
    # 二维与三维数据集来说, 高维欧式距离并没有太大的问题, 但对图片来说, 可能会出现高维诅咒, 
    # 想象一个马蹄形的空间, 里面三个坐标的位置欧式距离仅仅是距离! 因而改用了SSIM算法代替高维欧氏距离!
    showImage()
    getDist()