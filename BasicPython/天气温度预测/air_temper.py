import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime

from sklearn import preprocessing


features = pd.read_csv("./air_temps.csv")
features = pd.get_dummies(features)
feature_list = list(features.columns)

# Dealing with the time data
years = features["year"]
months = features["month"]
days = features["day"]

# Convert time to standard format
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year,month,day in zip(years,months,days)]
dates = [datetime.datetime.strptime(date,"%Y-%m-%d") for date in dates]

# Plot the data
plt.style.use("fivethirtyeight")
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (10,10))
fig.autofmt_xdate(rotation = 45)

# Plot the 4 subplots
ax1.plot(dates,features["actual"])
ax1.set_xlabel("")
ax1.set_ylabel("Temperatures")
ax1.set_title("Max Temperature")

ax2.plot(dates,features["temp_1"]) # Yesterday
ax2.set_xlabel("")
ax2.set_ylabel("Temperatures")
ax2.set_title("Previous Max Temperature")

ax3.plot(dates,features["temp_2"]) # The day before yesterday
ax3.set_xlabel("Date")
ax3.set_ylabel("Temperatures")
ax3.set_title("Two Days Prior Max Temperature")

ax4.plot(dates,features["friend"])
ax4.set_xlabel("Date")
ax4.set_ylabel("Temperatures")
ax4.set_title("Frined Estimate Max Temperature")

plt.tight_layout(pad = 2)
plt.show()

# Separate features and labels, then store the feature names
labels = np.array(features["actual"])
features = features.drop(["actual"], axis = 1)


# Look at the shape of the data
print(features.head(5))

# The following part is the content related to neural network!
# Feature standardization
input_features = preprocessing.StandardScaler().fit_transform(features)
print("Check input features = ", input_features)

x = torch.tensor(input_features, dtype = torch.float)
y = torch.tensor(labels, dtype = torch.float)


# Init the weight parameters
weights1 = torch.randn((14, 128), dtype = torch.float, requires_grad = True)
biases1 = torch.randn(128, dtype = torch.float, requires_grad = True)


weights2 = torch.randn((128, 1), dtype = torch.float, requires_grad = True)
biases2 = torch.randn(1, dtype = torch.float, requires_grad = True)


learning_rate = 0.001
losses = []


if __name__ == '__main__':
    # Gradient descent
    for i in range(1000):
        ## method mm is matrix multiplication
        hidden = x.mm(weights1) + biases1
        hidden = torch.relu(hidden)

        prediction = hidden.mm(weights2) + biases2

        ## Using MSE as the loss function
        loss = torch.mean(torch.pow(prediction - y, 2))
        losses.append(loss.data.numpy())

        if i % 100 == 0:
            print("Loss:", loss)
        loss.backward()

        ## Update the parameters
        weights1.data.add_(-learning_rate * weights1.grad.data)
        biases1.data.add_(-learning_rate * biases1.grad.data)
        weights2.data.add_(-learning_rate * weights2.grad.data)
        biases2.data.add_(-learning_rate * biases2.grad.data)

        ## Remember to reset the gradient every iteration
        weights1.grad.data.zero_()
        biases1.grad.data.zero_()
        weights2.grad.data.zero_()
        biases2.grad.data.zero_()

    # Observe the change of loss values
    plt.plot(losses)
    plt.show()
