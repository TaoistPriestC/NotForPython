import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

digits = load_digits()

print("digits.data: \n", digits.data)
print("digits.target: \n", digits.target)

train_x = digits.data
train_y = digits.target
epsilon = 0.01

def getImage(x, n):
    ret = list(x[np.random.choice(len(x),n)])
    for i,row in enumerate(ret):
        ret[i] = row.reshape(8,8)
    return np.array(ret)

def getDistb(k, m, n):
    ret = []
    for _ in range(k):
        ret.append(np.random.rand(m, n))
    return np.array(ret)


if __name__ == '__main__':
    src = getImage(train_x, 3)
    dis = getDistb(3, 8, 8)
    plt.title("Image Superimposed with Random Noise")
    plt.subplots_adjust(wspace = 0.35, hspace = 0.5)
    for i in range(1,10):
        plt.subplot(3, 3, i)
        if 1 <= i <= 3:
            plt.imshow(src[i - 1])
        if 4 <= i <= 6:
            plt.imshow(dis[i - 3 - 1])
        if 7 <= i <= 9:
            plt.imshow(src[i - 6 - 1] + epsilon * dis[i - 6 - 1])
    plt.show()