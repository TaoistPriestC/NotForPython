import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_value = [i for i in range(11)]
x_train = np.array(x_value, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_value = [2*i + 1 for i in x_value]
y_train = np.array(y_value, dtype=np.float32)
y_train = y_train.reshape(-1, 1)


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return out 


if __name__ == '__main__':
    ## Basic info of the model
    input_dim = 1
    output_dim = 1

    ## Model structure
    model = LinearRegression(input_dim, output_dim)
    
    ## Once CUDA is started, we will need do some CPU-GPU conversion!
    device = torch.device("cuda:0"  if torch.cuda.is_available() else "cpu")
    model.to(device)

    ## Hyperparameter 
    epoches = 1000
    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    criterion = nn.MSELoss()


    ## Training the model
    for ep in range(epoches):
        inputs = torch.from_numpy(x_train).to(device)
        labels = torch.from_numpy(y_train).to(device)

        ### Reset the gradient
        optimizer.zero_grad() 

        ### orward propagation
        outputs = model(inputs).to(device)

        ### Calculate loss value      
        loss = criterion(outputs, labels)
        loss.backward()

        ### backward propagation
        optimizer.step()

        if (ep + 1) % 50 == 0:
            print("epoch = {}, loss = {}".format(ep + 1, loss.item()))

    ## Using model.load_state_dict() to load the pkl file,
    ## Only training once then the model can be reused!

    torch.save(model.state_dict(), "./pkl/LinearRegression.pkl")


    ## Once you activate CUDA, you must pay attention to CPU-GPU conversion! 
    predicted = model(torch.from_numpy(x_train).to(device).requires_grad_()).data.cpu().numpy()
    
    ## Print the predictred value and real value 
    print(predicted)
    print(y_train)

    plt.plot(x_train, predicted)
    plt.plot(x_train, y_train)
    plt.show()

# There are many models have been implemented on pytorch hub
# https://github.com/pytorch/hub
# https://pytorch.org/hub/research-models
# 