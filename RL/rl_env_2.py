import numpy as np
import gymnasium as gym
from gymnasium import spaces
from RL.surrogate import geometric_reshape
from calc_init_param import calc_next_param
from core.param_util.param_tools import gen_param_csv
from automation import run_cmd
import os
import pandas as pd
import torch
import shutil
from data_collection import stress_collection_script, springback_collection_script
import subprocess

strip_length = 40
pre_length = 0.1
k = 0.05

class bending_env(gym.Env):
    def __init__(self, k=0.0, episode=0):

        # Define the state space size
        self.state_space = spaces.Box(low=-50000, high=50000, shape=(3, ), dtype=np.float64)

        # Define the action space size
        self.action_space = spaces.Discrete(n=15, start=1)

        self.k = k
        self.num_step = None
        self.state = None
        self.stress_dist = None

        self.pre_idx = None

        # Bending Parameter list
        self.param_list = []  # To be considered: add the pre-stretch parameter to it

        self.action_list = []  # Record series of actions for each episode for future use

        self.k_list = []

        self.max_step = 10  # Max number of bending steps
        if episode == None:
            self.mould_name = "test1"
        else:
            self.num_episode = episode
            self.mould_name = "test1_ep" + str(self.num_episode)

        # Some useful data path
        self.data_path_2 = "/Optimizing_benidng_parameter/data/mould_output/" + self.mould_name
        self.data_path_1 = "/Optimizing_bending_parameter/data/model/" + self.mould_name

    def reset(self):
        '''
            Reset the environment
        '''
        self.num_episode += 1
        self.mould_name = "test1_ep" + str(self.num_episode)
        self.num_step = 0
        self.data_path_2 = "/Optimizing_bending_parameter/data/mould_output/" + self.mould_name
        self.data_path_1 = "/Optimizing_bending_parameter/data/model/" + self.mould_name
        # Generate curve and mould for this episode
        print(self.mould_name)
        if not os.path.exists(self.data_path_2):
            os.makedirs(self.data_path_2)
        if not os.path.exists(self.data_path_1):
            os.makedirs(self.data_path_1)
        
        shutil.copy('./data/mould_output/test1/mould.stp', self.data_path_1)

        
        # Since the pre-stretch steps are all the same for each test, we simply used the one of test 0.
        
        self.rec = geometric_reshape()
        
        # geometric_position(self.rec, x)

        self.action_list = []  # Empty the action series
        self.k_list = []
        self.param_list = []  # Empty the param list
        self.pre_idx = None  # Reset pre_idx
        

        # Perform the pre-stretch step
        next_param, self.pre_idx = calc_next_param("\Optimizing_bending_parameter\data\mould_output\\" + "test1", 0, 
                                                    strip_length, 
                                                    pre_length, 
                                                    self.k, 
                                                    self.pre_idx)
        self.param_list.append(next_param)

        rel_param_list = gen_param_csv(
            param_list=self.param_list,
            output_path=self.data_path_2,
            pre_length=0.1,
            version="base",
        )

        rel_param = rel_param_list[-1]

        self.state = torch.tensor(rel_param[:2] + [rel_param[5]], dtype=torch.float32) * 1000 + 1e-6


        cmd = ['python ', 'gen_abaqus_model_step.py', self.mould_name, str(0)]
        run_cmd(cmd)

        step_name = "Step-" + str(self.num_step)
        recursion_path = os.path.abspath(self.data_path_1)
        p = subprocess.Popen(
        ["cmd", "/c", "abaqus", "cae", f"noGUI={os.path.join(recursion_path, 'script_create_model_{}'.format(step_name))}"],
        cwd=recursion_path,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
        try:
            outs, errs = p.communicate(timeout=2400)
            # print(outs.decode("utf-8"))
            # print(errs.decode("utf-8"))
            print("abaqus脚本执行成功")
        except TimeoutError:
            p.kill()
            print("abaqus脚本运行失败，请检查代码")
            exit(-1)


        # Extract stress distribution from ODB
        stress_collection_script(data_path=self.data_path_1+"/simulation/", mould_name=self.mould_name, step=0)
        self.stress_dist = self.stress_extract()

        return self.state

    def step(self, action, k):
        
        self.k = k
        self.k_list.append(self.k)
        # print(self.state)

        # remain_len = 1999 - self.pre_idx
        # pre_idx = self.pre_idx
        # Adding the next parameter to the list
        next_param, self.pre_idx = calc_next_param("\Optimizng_bending_parameter\data\mould_output\\" + "test1", action, strip_length, pre_length, self.k, self.pre_idx)
        
        if self.pre_idx == 1960:
            springback = self.get_springback()
            done = True
            return self.state, None, None, None, done, springback
        else:
            pass
        
        # forward_len = self.pre_idx - pre_idx
        self.action_list.append(action)
        self.param_list.append(next_param)

        self.num_step += 1

        # Execute the given action and return the next state, reward, and whether the episode is done

        # Transfer to relative parameter and store it in a csv
        rel_param_list = gen_param_csv(
            param_list=self.param_list,
            output_path=self.data_path_2,
            pre_length=0.1,
            version="base",
        )

        rel_param = rel_param_list[-1]

        self.state = torch.tensor(rel_param[:2] + [rel_param[5]], dtype=torch.float32) * 1000 + 1e-6

        cmd = ['python ', 'gen_abaqus_model_step.py', self.mould_name, str(self.num_step)]
        run_cmd(cmd)

        step_name = "Step-" + str(self.num_step)
        recursion_path = os.path.abspath(self.data_path_1)
        p = subprocess.Popen(
        ["cmd", "/c", "abaqus", "cae", f"noGUI={os.path.join(recursion_path, 'script_create_model_{}'.format(step_name))}"],
        cwd=recursion_path,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
        
        try:
            outs, errs = p.communicate(timeout=2400)
            # print(outs.decode("utf-8"))
            # print(errs.decode("utf-8"))
            print("abaqus脚本执行成功")
        except TimeoutError:
            p.kill()
            print("abaqus脚本运行失败，请检查代码")
            exit(-1)


        stress_collection_script(data_path=self.data_path_1+"/simulation/", mould_name=self.mould_name, step=self.num_step)

        # Check if the episode is done and calculate the reward
        reward_1, reward_2, reward_3 = self.calculate_reward()
        # print(reward)

        if self.pre_idx == 1959: 
            # remain_len = 0 
            springback = self.get_springback()
            done = True
        else:
            springback = 0
            done = False

        return self.state, reward_1, reward_2, reward_3, done, springback
    
    def calculate_reward(self):
        '''
            stress_dist: np array
            Reward is given by the sum of increase of stress on each element (for now)
            better representation?
        '''
        pre_stress_dist = self.stress_dist
        self.stress_dist = self.stress_extract()
        reward_1 = np.sum(self.stress_dist - pre_stress_dist) / len(self.stress_dist)
        reward_2 = np.max(self.stress_dist) - np.max(pre_stress_dist)
        reward_3 = np.var(self.stress_dist) - np.var(pre_stress_dist)
        return reward_1, reward_2, reward_3

    def get_springback(self):

        # Save action series to a csv
        action_list = pd.DataFrame({"action": self.action_list, "k": self.k_list})
        print(action_list)
        action_list.to_csv(self.data_path_1 + "/action_list.csv")

        print(self.param_list)
        rel_param_list = gen_param_csv(
            param_list=self.param_list,
            output_path=self.data_path_2,
            pre_length=0.1,
            version="base",
        )

        cmd = ['python ', 'gen_spring_back_model.py', self.mould_name, str(self.num_step)]
        run_cmd(cmd)
        
        springback_collection_script(self.data_path_1+"/simulation/", self.mould_name)

        springback_path = self.data_path_1 + "/simulation/springback_output.csv" 
        springback = pd.read_csv(springback_path)["Springback"].to_numpy()
        score = - np.max(springback) * 10  # Score is the max springback deviation, so the less the better
        
        return score
    
    def stress_extract(self):

        '''
            Extract stress distribution from the pre-stored csv
        '''

        csv_path = "/Optimizing_bending_parameter/data/model/{mould}/simulation/strip_mises_{step}.csv".format(
            mould = self.mould_name,
            step = "Step-" + str(self.num_step)
        )
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            x = df["S_Mises"].to_numpy()
            
            # x = torch.tensor(x, dtype=torch.float32)
        return x

if __name__ == "__main__":
    env = bending_env(episode=None)
    cum_reward_1 = 0
    cum_reward_2 = 0
    cum_reward_3 = 0
    env.reset()
    done = False
    while not done:
        print("You are currently at: {}".format(env.state))
        # action = float(input("Choose your next action"))
        # action = np.random.randint(1, 10)
        action = 8
        state, reward_1, reward_2, reward_3, remain_len, done, springback = env.step(action)
        print("Stress cumulated in this step is {} and {}".format(reward_1, reward_2))
        cum_reward_1 += reward_1.item()
        cum_reward_2 += reward_2.item()
        cum_reward_3 += reward_3.item()
