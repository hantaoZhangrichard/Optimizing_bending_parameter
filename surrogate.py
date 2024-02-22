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


class SurrogateNet_mlp(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SurrogateNet_mlp, self).__init__()
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

class SurrogateNet_cnn(nn.Module):
    def __init__(self):
        super(SurrogateNet_cnn, self).__init__()
        # Define convolutional layers for processing the current state (volumetric data)
        self.conv_module = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        # Define fully connected layers for processing the change vector
        self.fc_change = nn.Sequential(
            nn.Linear(3, 64),  # Assuming the change vector is a vector of length 3
            nn.ReLU()
        )
        # Define the final fully connected layer for predicting the future state
        self.fc_future_state = nn.Linear(32 * 9 * 1 * 1 + 64, 32 * 9 * 1 * 1)  

    def forward(self, x_current_state, x_change_vector):
        # Process current state (volumetric data) through conv_module
        x_current_state = self.conv_module(x_current_state)
        x_current_state = x_current_state.view(-1, 32 * 9 * 1 * 1)  # Flatten the output
        
        # Process change vector through fc_change
        x_change_vector = self.fc_change(x_change_vector)
        
        # Concatenate the outputs of conv_module and fc_change
        x = torch.cat((x_current_state, x_change_vector), dim=1)
        
        # Predict the future state using fc_future_state
        future_state = self.fc_future_state(x)
        return future_state

class SurrogateNet_multiMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SurrogateNet_multiMLP, self).__init__()
        self.stress_module = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.parameter_module = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        self.predict_module = nn.Sequential(
            nn.Linear(128 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x_current_state, x_parameter):
        x_current_state = self.stress_module(x_current_state)
        print(x_current_state.shape)
        x_parameter = self.parameter_module(x_parameter)
        x = torch.cat((x_current_state, x_parameter), dim=0)
        future_state = self.predict_module(x)
        return future_state

def load_data():
    path_x = "/Xie_and_Zhang/dataset_X.csv"
    path_y = "/Xie_and_Zhang/dataset_Y.csv"
    df_x = pd.read_csv(path_x, header=None)
    df_y = pd.read_csv(path_y, header=None)
    x_stress_tensor = torch.tensor(df_x.values[0:1512], dtype=torch.double)
    x_parameter_tensor = torch.tensor(df_x.values[1512:1515], dtype=torch.double)
    y_tensor = torch.tensor(df_y.values, dtype=torch.double)
    return x_stress_tensor, x_parameter_tensor, y_tensor

def train_one_epoch(epoch_index):

    running_loss = 0
    last_loss = 0
    for i, data in enumerate(train_loader):
            stress_inputs, p_inputs, labels = data
            optimizer.zero_grad()
            outputs = model(stress_inputs.double(), p_inputs.double())
            loss = loss_f(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i % 20 == 19:
                last_loss = running_loss/20
                print("Batch {} loss: {}".format(i+1, last_loss))
                running_loss = 0
    return last_loss

def visualize(outputs, labels, rec):
    labels = labels.numpy()
    outputs = outputs.numpy()
    labels = geometric_position(rec, labels)
    outputs = geometric_position(rec, outputs)
    # print(labels.shape)
    # print(labels[labels < 30])

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(labels[:, :, 0], cmap="rainbow", annot=False, vmin=80, vmax=200)
    plt.title('Label Heatmap')
    plt.xlabel('Data Point')
    plt.ylabel('Values')

    plt.subplot(1, 2, 2)
    sns.heatmap(outputs[:, :, 0], cmap="rainbow", annot=False, vmin=80, vmax=200)
    plt.title('Output Heatmap')
    plt.xlabel('Data Point')
    plt.ylabel('Values')

    plt.tight_layout()
    plt.show()

def geometric_reshape(mould_name):
    # Reconstruct the geometry of aluminium from coordinates
    csv_path = "/Xie_and_Zhang/data/model/{}/simulation/".format(mould_name) + "strip_mises_" + "Step-0" + ".csv"
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="Orig.X")
    # print(df["Orig.X"])
    x = df["Orig.X"].to_numpy()
    coor_x, counts_x = np.unique(x, return_counts=True)
    rec = np.empty(shape=(72, 7, 3)) # Shape of the aluminium
    x_i = 0
    near_x = [] # store nodes with coordinate very close to each other
    near_x_count = 0
    for x_value, x_count in zip(coor_x, counts_x):
        
        y_i = 0
        if x_count < 21 and near_x_count < 14:
            # Need to make sure every face with the same x coor has exactly 21 nodes
            near_x.append(x_value)
            near_x_count += x_count
            continue
        elif x_count < 21 and near_x_count == 14:
            near_x.append(x_value)
            face = df[df["Orig.X"].isin(near_x)]
            
            near_x = []
            near_x_count = 0
        else: 
            face = df[df["Orig.X"]==x_value]

        face = face.sort_values(by="Orig.Y")
        y = face["Orig.Y"].to_numpy()
        coor_y, counts_y = np.unique(y, return_counts=True)
        near_y = []
        near_y_count = 0
        for y_value, y_count in zip(coor_y, counts_y):
            if y_count < 3 and near_y_count < 2: # Need to make sure every line with the same x and y coor has exactly 3 nodes
                near_y.append(y_value)
                near_y_count += y_count
                continue
            elif y_count < 3 and near_y_count == 2:
                near_y.append(y_value)
                line = face[face["Orig.Y"].isin(near_y)]
                
                near_y = []
                near_y_count = 0
            else: 
                line = face[face["Orig.Y"]==y_value]
            
            line = line.sort_values(by="Orig.Z")
            
            rec[x_i, y_i] = line["Node_ID"].to_numpy()
            y_i += 1
        x_i += 1
    # Shape: 72*7*3
    return rec

def geometric_position(rec, data):
    # Put stress value into the aluminium tensor 
    result = np.empty(shape = rec.shape)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            for k in range(result.shape[2]):
                id = int(rec[i, j, k])
                if data.dim() > 1:
                    value = data[1, id-1]
                else: 
                    value = data[id-1]
                result[i, j, k] = value
    # print(result)
    return result


    
    


if __name__ == "__main__":

    mould_name = "test6"
    X_stress, X_parameter, y = load_data()
    X_stress = X_stress.t()
    X_parameter = X_parameter.t()
    y = y.t()
    # print(X)
    print(y[1][1])
    X_train, X_val, p_train, p_val, y_train, y_val = train_test_split(X_stress, X_parameter, y, test_size = 0.2, random_state = 42)
    print(X_val.shape)

    rec = geometric_reshape(mould_name)

    train_dataset = TensorDataset(X_train, p_train, y_train)
    val_dataset = TensorDataset(X_val, p_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size = 4, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 4, shuffle = False)

    input_dim = X_stress.shape[1]
    output_dim = y.shape[1]
    model = SurrogateNet_multiMLP(input_dim, output_dim).double()

    loss_f = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 200
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
                vinputs_stress, vinputs_p, vlabels = vdata
                voutputs = model(vinputs_stress, vinputs_p)
                vloss = loss_f(voutputs, vlabels)
                running_vloss += vloss
                if epoch > 0.95 * num_epochs and i % 6 == 0:
                    visualize(voutputs, vlabels, rec)

        avg_vloss = running_vloss / (i+1)

        print("Loss train:{} validation:{}".format(avg_loss, avg_vloss))

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
    print("Best_vloss:{}".format(best_vloss))

    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    # Add any other information you want to save
    }, 'Surrogate_model.pth')
