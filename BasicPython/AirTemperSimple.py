import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime

from sklearn import preprocessing


features = pd.read_csv("./AirTemps.csv")
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
# plt.style.use("fivethirtyeight")
# fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (10,10))
# fig.autofmt_xdate(rotation = 45)

# ax1.plot(dates,features["actual"])
# ax1.set_xlabel("")
# ax1.set_ylabel("Temperatures")
# ax1.set_title("Max Temperature")

# ax2.plot(dates,features["temp_1"]) # Yesterday
# ax2.set_xlabel("")
# ax2.set_ylabel("Temperatures")
# ax2.set_title("Previous Max Temperature")

# ax3.plot(dates,features["temp_2"]) # The day before yesterday
# ax3.set_xlabel("Date")
# ax3.set_ylabel("Temperatures")
# ax3.set_title("Two Days Prior Max Temperature")

# ax4.plot(dates,features["friend"])
# ax4.set_xlabel("Date")
# ax4.set_ylabel("Temperatures")
# ax4.set_title("Frined Estimate Max Temperature")

# plt.tight_layout(pad = 2)
# plt.show()

# Separate features and labels, then store the feature names
labels = np.array(features["actual"])
features = features.drop(["actual"], axis = 1)

# Look at the shape of the data
print(features.head(5))


# The following part is the content related to neural network!
# Feature standardization
input_features = preprocessing.StandardScaler().fit_transform(features)


x = torch.tensor(input_features, dtype = torch.float)
y = torch.tensor(labels, dtype = torch.float)

# Init the weight parameters
weights1 = torch.randn((14, 128), dtype = torch.float, requires_grad = True)
biases1 = torch.randn(128, dtype = torch.float, requires_grad = True)

weights2 = torch.randn((128, 1), dtype = torch.float, requires_grad = True)
biases2 = torch.randn(1, dtype = torch.float, requires_grad = True)

learning_rate = 0.001
losses = []


# Setting hyperparamters and parameter
input_size = input_features.shape[1]
hidden_size = 128
output_size = 1
batch_size = 16

myNN = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.Sigmoid(), 
    torch.nn.Linear(hidden_size, output_size)
)

cost = torch.nn.MSELoss(reduction = "mean")
optimizer = torch.optim.Adam(myNN.parameters(), lr = 0.001)

losses = []

if __name__ == '__main__':
    ## Using MBGD(Mini-batch gradiant descent) method to optimize the model
    try:
        ### Try to load the pkl that used to set paramters
        myNN.load_state_dict(torch.load("AirTemper.pkl"))

    except FileNotFoundError as e:
        ### Train it again if there is no pkl file
        data_size = len(input_features)
        for i in range(1000):
            batch_loss = []
            for start in range(0, data_size, batch_size):
                end = start + batch_size if start + batch_size < data_size else data_size
                xx = torch.tensor(input_features[start:end], dtype = torch.float, requires_grad = True)
                yy = torch.tensor(labels[start:end], dtype = torch.float, requires_grad = True)
                
                prediction = myNN(xx)
                prediction = prediction.squeeze(-1)

                loss = cost(prediction, yy)
                optimizer.zero_grad()
                
                loss.backward(retain_graph = True)
                optimizer.step()

                batch_loss.append(loss.data.numpy())

            if i % 100 == 0:
                losses.append(np.mean(batch_loss))
                print("the {:0>3d}-th epoch : {:5f}".format(i, np.mean(batch_loss)))
        
        ### Save the model
        torch.save(myNN.state_dict(),"AirTemper.pkl")

    finally:
        ### Display the difference between real value and predict value
        predict = myNN(x).data.numpy()
        real_data = pd.DataFrame(data = {"date":dates, "actual":labels})
        predict_data = pd.DataFrame(data = {"date":dates, "predict":predict.reshape(-1)})

        plt.plot(real_data["date"], real_data["actual"], "b-", label = "actual")
        plt.plot(predict_data["date"], predict_data["predict"], "ro", label = "predict")

        plt.xticks(fontsize = 8, rotation = 45)
        plt.yticks(fontsize = 8)
        plt.legend()

        plt.xlabel("Date")
        plt.ylabel("Maximun Temperature(F)")
        plt.title("Actual and predicted Values")
        plt.rcParams['figure.figsize'] = (12.0, 8.0)
        plt.show()
