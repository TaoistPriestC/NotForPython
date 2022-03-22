import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader,TensorDataset,SubsetRandomSampler


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

model = AutoEncoder()
model.load_state_dict(torch.load("../models/auto_encoder.pkl"))


np.random.seed(1)
torch.manual_seed(1)

# 抽取二十五分子图查看
# cnt = 0
# rand_noise_outputs = []
# for i in range(25):
#     cnt += 1
#     plt.subplot(5, 5, cnt)
#     tmp_output = model.decoder(torch.Tensor(np.random.randn(3))).detach().numpy()
#     tmp_output = tmp_output.reshape(28, 28)
#     plt.imshow(tmp_output, cmap = "gray")
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

# 目标是把整个MNIST数据集都变成三维数据
mnist = datasets.MNIST(
    root = "../data",
    train = True,
    transform = transforms.ToTensor(),
    download = False
)

# print(mnist.train_data.shape)
# print(mnist.test_data.shape)

features, labels = [], []
for i, (image,label) in enumerate(mnist):
    print("Sample: {:>05d}".format(i))
    
    image = list(model.encoder(image.view(-1, 28 * 28)).detach().numpy().reshape(3))
    label = np.array(label, dtype = np.uint8)
    
    features.append(image)
    labels.append(label)


print(len(features))
print(len(labels))

#=============================================#
ds = pd.DataFrame({
    "x1" : [x[0] for x in features],
    "x2" : [x[1] for x in features],
    "x3" : [x[2] for x in features],
    "labels":labels
})
ds.to_csv("../data/DIY/ae_mnist3d.csv")
print(ds)

