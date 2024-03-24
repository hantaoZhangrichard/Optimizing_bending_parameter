import numpy as np
import gym
from gym import spaces
from surrogate import SurrogateNet_multiMLP, geometric_position, geometric_reshape
from calc_init_param import calc_next_idx, calc_next_param
from core.param_util.param_tools import gen_param_csv
from automation import run_cmd
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import subprocess

# Define the Actor network
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the Critic network
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class bending_env(gym.Env):
    def __init__(self, episode=0):

        # Define the state space size
        self.state_space = spaces.Box(low=0, high=200, shape=(72, 7, 3), dtype=np.double)

        # Define the action space size
        self.action_space = spaces.Discrete(n=15, start=1)

        # Initialize the current state with the stress distribution after pre-stretch
        self.state = None

        self.pre_idx = None
        self.pre_param = [321.1,0.0,0.0,0,-0.0,0.0]  # Pre-stretch length

        # Surrogate model
        self.model = SurrogateNet_multiMLP(1512, 1512)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        checkpoint = torch.load("C:\Optimizing_bending_parameter\Surrogate_model.pth")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Bending Parameter list
        self.param_list = []  # To be considered: add the pre-stretch parameter to it

        self.action_list = []  # Record series of actions for each episode for future use

        self.max_step = 10  # Max number of bending steps

        self.num_episode = episode

        self.mould_name = "test" + str(self.num_episode)

        # Some useful data path
        self.data_path_2 = "./data/mould_output/" + self.mould_name
        self.data_path_1 = "./data/model/" + self.mould_name

    def reset(self):
        # Reset the environment
        self.num_episode += 1
        self.mould_name = "test" + str(self.num_episode)
        self.data_path_2 = "./data/mould_output/" + self.mould_name
        self.data_path_1 = "./data/model/" + self.mould_name
        # Generate curve and mould for this episode
        print(self.mould_name)
        if not os.path.exists(self.data_path_2):
            os.makedirs(self.data_path_2)
        if not os.path.exists(self.data_path_1):
            os.makedirs(self.data_path_1)
        cmd = ['python ', 'gen_curve_and_mould.py', self.mould_name]
        # print(cmd)
        
        run_cmd(cmd)
        shutil.copy(self.data_path_2 + '/mould.stp', self.data_path_1)

        '''
            Initialize the state with the stress distribution after pre-stretch.
            Since the pre-strech steps are all the same for each test, we simply used the one of test 0.
        '''
        self.rec = geometric_reshape(self.mould_name)
        csv_path = "/Optimizing_bending_parameter/data/model/test0/simulation/strip_mises_Step-0.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            x = df["S_Mises"]
            x = torch.tensor(x, dtype=torch.float32)
        self.state = x
        # geometric_position(self.rec, x)

        self.action_list = []  # Empty the action series
        self.param_list = []  # Empty the param list
        self.pre_idx = None  # Reset pre_idx
        return self.state

    def step(self, action):
        self.action_list.append(action)
        # print(self.state)
        strip_length = 40
        pre_length = 0.1
        k = 0.05 
        # Adding the next parameter to the list
        next_param, self.pre_idx = calc_next_param("./data/mould_output/" + self.mould_name, action, strip_length, pre_length, k, self.pre_idx)
        self.param_list.append(next_param)
        t = (np.array(next_param - np.array(self.pre_idx))).tolist()
        t = torch.tensor(t[:2] + [t[5]], dtype=torch.float32)
        # Execute the given action and return the next state, reward, and whether the episode is done
        self.model.eval()

        self.state = self.model(self.state, t)  # Surrogate model as transition function
        
        # Check if the episode is done and calculate the reward
        if self.pre_idx == 1999:
            reward = self.calculate_reward()
            done = True
        else:
            reward = 0
            done = False  
        return self.state, reward, done, {}

    def calculate_reward(self):

        # Save action series to a csv
        action_list = pd.DataFrame(self.action_list)
        action_list.to_csv(self.data_path_1 + "action_list.csv")

        # Calculate the reward based on the current state
        if not os.path.exists(self.data_path_1):
            os.makedirs(self.data_path_1)
        if not os.path.exists(self.data_path_2):
            os.makedirs(self.data_path_2)
        print(self.param_list)
        rel_param_list = gen_param_csv(
            param_list=self.param_list,
            output_path=self.data_path_2,
            pre_length=0.1,
            version="base",
        )

        tasks = ['gen_abaqus_model.py', 'gen_spring_back_model.py', 'data_collection.py']

        for i in range(len(tasks)):
            cmd = ['python ', tasks[i], self.mould_name]
            run_cmd(cmd)
            
        springback_path = self.data_path_1 + "/simulation/springback_output.csv" 
        springback = pd.read_csv(springback_path)["Springback"]
        reward = max(springback.tolist()) * 10 # Reward is the max springback deviation, so the less the better
        # print(reward)
        return reward

if __name__ == "__main__":

    env = bending_env(episode=1)
    initial_state = env.reset()
    while True:
        action = np.random.randint(1, 5)
        state, reward, done, _ = env.step(action)
        if done == True:
            break
