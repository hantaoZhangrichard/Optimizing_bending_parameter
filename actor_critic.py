import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, input_dim, n_actions, alpha):
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        # Define the final fully connected layer for generating the action
        self.fc1 = nn.Linear(self.input_dim, 256)  
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.n_actions)
        self.relu = nn.ReLU()

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, x): 
        x = torch.tensor(x)
        # Fully connected layers with ReLU activation
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # print(self.fc3(x))
        probs = F.softmax((self.fc3(x)), dim=0)
        # print(probs)
        return probs

# Define the Critic network
class Critic(nn.Module):
    def __init__(self, input_dim, alpha):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        # Define the final fully connected layer for generating the action
        self.fc1 = nn.Linear(self.input_dim, 256)  
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, x):
        x = torch.tensor(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent():
    def __init__(self, input_dim, n_actions, gamma=0.99, lr=1e-1):
        super(Agent, self).__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.actor = Actor(self.input_dim, self.n_actions, alpha=lr)
        self.critic = Critic(self.input_dim, alpha=lr)
        self.gamma = gamma
        self.log_probs = None
        
    def choose_action(self, observation):
        # x = torch.tensor(observation)
        probs = self.actor.forward(observation)
        action_probs = torch.distributions.Categorical(probs)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)
        print("Action is {}".format(action))
        return action.item()
        
    def learn(self, state, new_state, reward, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        value_1 = self.critic.forward(state)
        # print(value_1)
        value_2 = self.critic.forward(new_state)

        reward = torch.tensor(reward)

        delta = reward + self.gamma * value_2 * (1-int(done)) - value_1

        critic_loss = delta ** 2
        # print(critic_loss)
        actor_loss = -self.log_probs * delta
        # param = self.actor.parameters()

        (actor_loss + critic_loss).backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
            

if __name__ == "__main__":
    agent = Agent(input_dim=3, n_actions=10)
    observation = np.array([1, 2, 3], dtype=np.float32)
    action = agent.choose_action(observation)
    print(action)