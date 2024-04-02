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
from surrogate import geometric_reshape, geometric_position

class SurrogateNet_springback(nn.Module):
    def __init__(self, rec):
        super(SurrogateNet_springback, self).__init__()
        # Define convolutional layers for processing the current state (volumetric data)
        self.conv_module = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ).double()
       
        # Define the final fully connected layer for predicting the springback
        self.fc1 = nn.Linear(32 * 18 * 1 * 1, 64, dtype=torch.double)  
        self.fc2 = nn.Linear(64, 1, dtype=torch.double)
        self.rec = rec
        self.relu = nn.ReLU()
    def forward(self, stress):
        # Process current state (volumetric data) through conv_module
        # print(stress.shape)
        x = torch.tensor(geometric_position(self.rec, stress), dtype=torch.double)
        
        # print(stress)
        x = self.conv_module(x)
            
        x = x.view(-1, 32 * 18 * 1 * 1)  # Flatten the output
            
        # Fully connected layers with ReLU activation
        x = self.relu(self.fc1(x))
            
        springback = self.fc2(x)
        return springback

def load_data():
    path_x = "./dataset_springback_X.csv"
    path_y = "./dataset_springback_Y.csv"
    df_x = pd.read_csv(path_x, header=None)
    df_y = pd.read_csv(path_y, header=None)
    stress_tensor = torch.tensor(df_x.values[0:1512], dtype=torch.double)
    y_tensor = torch.tensor(df_y.values, dtype=torch.double)
    return stress_tensor, y_tensor

def train_one_epoch(epoch_index):

    running_loss = 0
    last_loss = 0
    for i, data in enumerate(train_loader):
            stress_inputs, labels = data
            optimizer.zero_grad()
            outputs = model(stress_inputs.double())
            # print(stress_inputs.shape)
            # print(outputs)
            # print(labels)
            loss = loss_f(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i % 20 == 19:
                last_loss = running_loss/20
                print("Batch {} loss: {}".format(i+1, last_loss))
                running_loss = 0
    return last_loss

if __name__ == "__main__":
    mould_name = "test0"
    X_stress, y = load_data()
    # print(X_stress.shape)
    X_stress = X_stress.t()
    # y = y.t()

    X_train, X_val, y_train, y_val = train_test_split(X_stress, y, test_size = 0.2, random_state = 42)
    print(X_val.shape)
    
    rec = geometric_reshape()
    # print(rec)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    
    model = SurrogateNet_springback(rec).double()

    loss_f = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
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
                vinputs_stress, vlabels = vdata
                voutputs = model(vinputs_stress)
                vloss = loss_f(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i+1)

        print("Loss train:{} validation:{}".format(avg_loss, avg_vloss))

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
    print("Best_vloss:{}".format(best_vloss))

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Add any other information you want to save
    }, 'Surrogate_springback_model.pth')