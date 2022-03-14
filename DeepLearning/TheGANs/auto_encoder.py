import os
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader,TensorDataset,SubsetRandomSampler

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

## 设置随机数种子 
torch.manual_seed(1)
np.random.seed(1)

## 设置一些重要的超参数
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
NUM_TEST_IMG = 10

# datasets模块会把数据集与标签打包成一个元组的形式返回
# 然而需要注意, train_data每个子项是元组类型, train_data.data里面的每个子项都是张量类型
train_data = datasets.MNIST(
    root = "../data",
    train = True,
    transform = transforms.ToTensor(),
    download = False
)

loader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)

# 随机抽取几个样本用来展示
# 载入SubsetRandomSampler(indices),这个函数会按给定列表从数据集中按照下标取元素

samples_to_show = SubsetRandomSampler(np.random.choice(range(len(train_data)), NUM_TEST_IMG))

# 自动编码机是一个无监督模型,所以用不到labels

# cnt = 0 
# plt.subplots_adjust(wspace = 0.5, hspace = 1.0)
# plt.suptitle("Some smaples")
# for i in samples_to_show:
#     cnt += 1
#     plt.subplot(2, 5, cnt)
#     image, label = train_data[i]
#     image, label = image.reshape(28, 28), str(label)
     
#     plt.title(label)
#     plt.imshow(image, cmap = "gray")
# plt.show()

class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), 
            nn.Tanh(), 
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

Coder = AutoEncoder()
optimizer = optim.Adam(Coder.parameters(), lr = LR)
criterion = nn.MSELoss()

print(Coder)


def train():
    losses = []
    for ep in range(EPOCH):
        for i, (image,label) in enumerate(loader):
            tx = image.view(-1, 28 * 28)
            ty = image.view(-1, 28 * 28)
            tl = label

            encoded, decoded = Coder(tx)

            ## 优化器梯度清零与模型梯度直接清零有何区别呢?
            optimizer.zero_grad()
            
            loss = criterion(decoded, ty)
            loss.backward()
            losses.append(loss.item())


            optimizer.step()
            if i % 5 == 0:
                print("Epoch: {}\t |  train_loss: {:.4f}".format(ep, loss.data))

    print("Training Finished!")
    torch.save(Coder.state_dict(), "../models/auto_encoder.pkl")
    return losses


def show_encoded_data(encoded_data):
    fig = plt.figure()
    fig.suptitle("降维之后的数据分布")
    ax = Axes3D(fig)
    X = encoded_data.data[:, 0].numpy()
    Y = encoded_data.data[:, 1].numpy()
    Z = encoded_data.data[:, 2].numpy()

    for x, y, z, s in zip(X, Y, Z, labels.numpy()):
        c = cm.rainbow(int(255 * s / 9))
        ax.text(x, y, z, s, backgroundcolor = c) 
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
    plt.show()


def show_decoded_data(decoded_data):
    plt.ion()
    plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
    fig, axes = plt.subplots(5, 5)
    fig.suptitle("解码数据展示")
    for i in range(0, len(decoded_data)):
        t = i % 25
        axes[t // 5][t % 5].set_xticks([])
        axes[t // 5][t % 5].set_yticks([])
        axes[t // 5][t % 5].imshow(decoded_data[i].detach().cpu().numpy().reshape(28,28), cmap = "gray")
    
        if (i + 1) % 25 == 0:
            fig.show()
            plt.pause(0.1)
    plt.show()
    plt.ioff()


def show_difference():
    plt.ion()
    for i in range(10):
        test_data = train_data[i][0].view(-1, 28 * 28)
        encoded_data, decoded_data = Coder(test_data)

        # print('输入的数据的维度', train_data.train_data[i].size())
        # print('输出的结果的维度', result.size())
        
        pred_result = decoded_data.view(28,28)
        
        ## 对于设置了需要计算梯度的张量是无法直接转为numpy数据的, 转化之前必须使用detach拆卸
        plt.figure(1, figsize = (10, 4))
        plt.subplot(121)
        plt.title("原始数据")
        plt.imshow(train_data[i][0].view(28,28).numpy(), cmap = "Greys")

        plt.figure(1, figsize = (10, 4))
        plt.subplot(122)
        plt.title("解码数据")
        plt.imshow(pred_result.detach().numpy(), cmap = "Greys")

        plt.show()
        plt.pause(0.5)
    plt.ioff()



if __name__ == "__main__":
    if not os.path.exists("../models/auto_encoder.pkl"):
        train()
    
    Coder.load_state_dict(torch.load("../models/auto_encoder.pkl"))
  
    ## 把元素列表拆为两个列表,
    ## 其中train_data与test_data两条属性已经预备弃用了, 可通过其获取uint8类型的图片,
    ## 然而直接通过索引会获取一个元组(X, Y) 并且图片X已经做好了归一化处理,
    data = torch.Tensor([train_data[i][0].numpy()  for i in range(200)]).float()
    labels = torch.Tensor([train_data[i][1]  for i in range(200)]).long()
    
    view_data = data.view(-1, 28 * 28)
    encoded_data, decoded_data = Coder(view_data)
    

    ## 配置中文环境
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    show_encoded_data(encoded_data)
    show_decoded_data(decoded_data)
    show_difference()

    

## 这份代码复现了下面这篇帖子:
## https://zhuanlan.zhihu.com/p/116769890


## 思考: transform发挥了什么作用?