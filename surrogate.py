import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import subprocess
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


class SurrogateNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SurrogateNet, self).__init__()
        self.inputLayer = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.hidden1 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.hidden2 = nn.Linear(256, 256)
        self.relu3 = nn.ReLU()
        self.hidden3 = nn.Linear(256, 512)
        self.relu4 = nn.ReLU()
        self.outputLayer = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.relu1(self.inputLayer(x))
        x = self.relu2(self.hidden1(x))
        x = self.relu3(self.hidden2(x))
        x = self.relu4(self.hidden3(x))
        x = self.outputLayer(x)
        return x

def load_data():
    path_x = "/Xie_and_Zhang/dataset_X.csv"
    path_y = "/Xie_and_Zhang/dataset_Y.csv"
    df_x = pd.read_csv(path_x, header=None)
    df_y = pd.read_csv(path_y, header=None)
    x_tensor = torch.tensor(df_x.values, dtype=torch.double)
    y_tensor = torch.tensor(df_y.values, dtype=torch.double)
    return x_tensor, y_tensor

def train_one_epoch(epoch_index):

    running_loss = 0
    last_loss = 0
    for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.double())
            loss = loss_f(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i % 20 == 19:
                last_loss = running_loss/20
                print("Batch {} loss: {}".format(i+1, last_loss))
                running_loss = 0
    return last_loss

def visualize(outputs, labels):
    labels = labels.numpy()
    outputs = outputs.numpy()
    # print(labels.shape)
    # print(labels[labels < 30])

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(labels.reshape(12, -1), cmap="rainbow", annot=False)
    plt.title('Label Heatmap')
    plt.xlabel('Data Point')
    plt.ylabel('Values')

    plt.subplot(1, 2, 2)
    sns.heatmap(outputs.reshape(12, -1), cmap="rainbow", annot=False)
    plt.title('Output Heatmap')
    plt.xlabel('Data Point')
    plt.ylabel('Values')

    plt.tight_layout()
    plt.show()

def geometric_reshape(test):
    mould_name = "test" + str(test)
    csv_path = "/Xie_and_Zhang/data/model/{}/simulation/".format(mould_name) + "strip_mises_" + "Step-0" + ".csv"
    df = pd.read_csv(csv_path)
    x_axis = {}
    # print(len(df))
    # while len(df) != 0:
    df = df.sort_values(by="Orig.X")
    # print(df["Orig.X"])
    x = df["Orig.X"].to_numpy()
    coor_x, counts_x = np.unique(x, return_counts=True)
    print(len(coor_x))
    rec = np.empty(shape=(72, 7, 3))
    jump_count = 0
    # print(coor_x)
    x_i = 0
    near_x = [] # store nodes with coordinate very close to each other
    near_x_count = []
    for x_value, x_count in zip(coor_x, counts_x):
        
        y_i = 0
        # print(len(coor_x))
        if x_count < 21 and sum(near_x_count) < 14:
            print(x_count) # Need to make sure every face with the same x coor has exactly 21 nodes
            near_x.append(x_value)
            near_x_count.append(x_count)
            jump_count += 1
            continue
        elif sum(near_x_count) == 21:
            face = df[df["Orig.X"].isin(near_x)]
            print(len(face))
            near_x = []
            near_x_count = []
        else: 
            face = df[df["Orig.X"]==x_value]

        face = face.sort_values(by="Orig.Y")
        y = face["Orig.Y"].to_numpy()
        coor_y, counts_y = np.unique(y, return_counts=True)
        near_y = []
        near_y_count = []
        for y_value, y_count in zip(coor_y, counts_y):
            if y_count < 3 and sum(near_y_count) < 2: # Need to make sure every face with the same x coor has exactly 21 nodes
                near_y.append(y_value)
                near_y_count.append(y_count)
                continue
            elif sum(near_y_count) == 3:
                line = face[face["Orig.Y"].isin(near_y)]
                print(line)
                near_y = []
                near_y_count = []
            else: 
                line = face[face["Orig.Y"]==y_value]
            print("Face: {}, Line: {}".format(x_i, y_i))
            
            # print("y_count", y_count)
            line = line.sort_values(by="Orig.Z")
            # print(line)
            rec[x_i, y_i] = line["Node_ID"].to_numpy()

            y_i += 1
        x_i += 1
    print(jump_count)
    print(rec)


    
    


if __name__ == "__main__":
    '''X, y = load_data()
    X = X.t()
    y = y.t()
    # print(X)
    print(y[1][1])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)
    print(X_val.shape)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size = 4, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 4, shuffle = False)

    input_dim = X.shape[1]
    output_dim = y.shape[1]
    model = SurrogateNet(input_dim, output_dim).double()

    loss_f = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 300
    best_vloss = 1000000
    for epoch in range(num_epochs):
        print("Epoch {}:".format(epoch + 1))
        model.train(True)

        avg_loss = train_one_epoch(epoch)

        running_vloss = 0
        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                # print(i)
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_f(voutputs, vlabels)
                running_vloss += vloss
                if epoch > 0.95 * num_epochs and i % 6 == 0:
                    visualize(voutputs, vlabels)

        avg_vloss = running_vloss / (i+1)

        print("Loss train:{} validation:{}".format(avg_loss, avg_vloss))

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
    print("Best_vloss:{}".format(best_vloss))'''
    geometric_reshape(6)

