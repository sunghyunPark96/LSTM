import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import Adam


class LSTM_dataset(Dataset):
    def __init__(self,x,y):
        super(LSTM_dataset,self).__init__()
        self.x = x
        self.y = y

        for key in self.x.columns:
            value_max = self.x[key].max()
            value_min = self.x[key].min()
            self.x[key] = (self.x[key]- value_min) / (value_max-value_min)

        self.y = ((self.y - self.y.min())/(self.y.max()-self.y.min()))

    def __len__(self):
        return len(self.y)-10

    def __getitem__(self, index):
        x = torch.tensor(self.x.iloc[index:index+10].values,dtype=torch.float32)
        y = torch.tensor(self.y.iloc[index+10],dtype=torch.float32)
        y = torch.unsqueeze(y,0)

        return x,y

class LSTM1(nn.Module):
  def __init__(self):
    super(LSTM1, self).__init__()
    self.lstm = nn.LSTM(input_size=4, hidden_size=2,
                      num_layers=1, batch_first=True) #lstm
    self.fc1 = nn.Linear(2*10, 128) #fully connected 1
    self.fc2 = nn.Linear(128, 64) #fully connected last layer
    self.fc3 = nn.Linear(64,32)
    self.fc4 = nn.Linear(32,1)
    self.relu = nn.ReLU()

  def forward(self,x):

    # Propagate input through LSTM
    x, hn = self.lstm(x) #lstm with input, hidden, and internal state
    x = torch.reshape(x,(x.shape[0],-1))
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    x = self.relu(x)
    x = self.fc4(x)
    return x

if __name__ == "__main__":
    data = pd.read_csv("C:/Users/MOBIS/Desktop/딥러닝 공부 자료/딥러닝 알고리즘 구현/LSTM/SBUX.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    data.drop(columns="Volume",inplace=True)

    x_data = data.iloc[:,:-1]
    y_data = data["Adj Close"]

    x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2)

    train_dataset = LSTM_dataset(x_train,y_train)
    test_dataset = LSTM_dataset(x_test,y_test)
    
    train_dataloader = DataLoader(train_dataset,shuffle=False,batch_size=32)
    test_dataloader = DataLoader(test_dataset,shuffle=False,batch_size=1)

    LSTM_model = LSTM1()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
    LSTM_model.to(device=device)

    epoch = 500
    lr = 0.0001
    optim = Adam(params=LSTM_model.parameters(),lr=lr)
    criterion = nn.MSELoss()
    save_path = "C:/Users/MOBIS/Desktop/딥러닝 공부 자료/딥러닝 알고리즘 구현/LSTM/model/LSTM.pt"


    signal = input(str("train : y test : n"))
    avg_loss = []  

    if signal == "y":
        for i in range(epoch):
            LSTM_model.train()
            epoch_loss = 0

            for data,label in train_dataloader:
                optim.zero_grad()
                pred = LSTM_model(data.to(device=device))
                loss = criterion(pred,label.to(device=device))
                batch_loss = loss.item()
                epoch_loss += batch_loss
                loss.backward()
                optim.step()

            avg_epoch_loss = epoch_loss / len(train_dataloader)
            avg_loss.append(avg_epoch_loss)

            print(f"epoch is {i+1}, avg_epoch_loss{avg_epoch_loss}")
        plt.plot(avg_loss)
        plt.show()
        torch.save(LSTM_model.state_dict(),save_path)

    if signal == "n":
        LSTM_model.eval()
        LSTM_model.load_state_dict(torch.load(save_path))

        preds = []
        with torch.no_grad():
            for data,label in test_dataloader:
                pred = LSTM_model(data.to(device=device))
                preds.append(pred.item())


        plt.plot(np.array(test_dataset.y.reset_index().drop(columns="Date")[10:]).squeeze(),label ="real")
        plt.plot(preds, label ="pred")
        plt.legend()
        plt.show()
