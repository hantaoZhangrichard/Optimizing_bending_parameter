import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

from surrogate import geometric_reshape, geometric_position

# Define the Actor network
class Actor(nn.Module):
    def __init__(self, rec):
        super(Actor, self).__init__()
        self.conv_module = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
       
        # Define the final fully connected layer for generating the action
        self.fc1 = nn.Linear(32 * 18 * 1 * 1, 64)  
        self.fc2 = nn.Linear(64, 1)
        self.rec = rec
        self.relu = nn.ReLU()
        self.sigmoid = nn.sigmoid()

    def forward(self, stress):
        x = torch.tensor(geometric_position(self.rec, stress[i,:]), dtype=torch.double)
        
        # print(stress)
        x = self.conv_module(x)
            
        x = x.view(-1, 32 * 18 * 1 * 1)  # Flatten the output
            
        # Fully connected layers with ReLU activation
        x = self.relu(self.fc1(x))
            
        action = self.sigmoid(self.fc2(x))
        return action

# Define the Critic network
class Critic(nn.Module):
    def __init__(self, rec):
        self.conv_module = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
       
        # Define the final fully connected layer for generating the action
        self.fc1 = nn.Linear(32 * 18 * 1 * 1, 64)  
        self.fc2 = nn.Linear(64+1, 64)
        self.fc3 = nn.Linear(64, 1)
        self.rec = rec
        self.relu = nn.ReLU()
        self.sigmoid = nn.sigmoid()

    def forward(self, stress, action):
        x = torch.tensor(geometric_position(self.rec, stress[i,:]), dtype=torch.double)
        # print(stress)
        x = self.conv_module(x)
        x = x.view(-1, 32 * 18 * 1 * 1)  # Flatten the output
        x = self.relu(self.fc1(x))
        x = torch.cat([x, action], 1)  # Check for concatenate dimension later
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
