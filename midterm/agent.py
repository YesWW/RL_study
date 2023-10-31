import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from collections import deque


class ReplayBuffer:
    '''
    saves transition datas to buffer
    '''

    def __init__(self, buffer_size=100000, n_step=1, gamma=0.85):
        '''
        Replay Buffer initialize function

        args:
            buffer_size: maximum size of buffer
            n_step: n step if using n step DQN
            gamma: discount factor for n step
        '''
        self.buffer_size = buffer_size
        self.n_step = n_step
        self.gamma = gamma
        
        self.states = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.next_states = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)

        self.n_states = deque(maxlen=self.n_step)
        self.n_actions = deque(maxlen=self.n_step)
        self.n_rewards = deque(maxlen=self.n_step)
        self.n_next_states = deque(maxlen=self.n_step)
        self.n_dones = deque(maxlen=self.n_step)


    def __len__(self) -> int:
        return len(self.states)


    def add(self, state, action, reward, next_state, done):
        '''
        add sample to the buffer
        '''
        
        if self.n_step > 1:
            self.n_states.append(state)
            self.n_actions.append(action)
            self.n_rewards.append(reward)
            self.n_next_states.append(next_state)
            self.n_dones.append(done)
            
            if len(self.n_states) == self.n_step:
                # append to main buffer by preprocessing n step
                n_step_reward = 0
                for i in range(self.n_step):
                    n_step_reward += self.gamma**i * self.n_rewards[i]
                    if self.n_dones[i]:
                        break
                self.states.append(self.n_states[0])
                self.actions.append(self.n_actions[0])
                self.rewards.append(n_step_reward)
                self.next_states.append(next_state)
                self.dones.append(self.n_dones[-1])
                #assert NotImplementedError
        else:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)

    
    def sample(self, batch_size, device=None):
        '''
        samples random batches from buffer

        args:
            batch_size: size of the minibatch
            device: pytorch device

        returns:
            states, actions, rewards, next_states, dones
        '''
        
        index = np.random.choice(range(len(self.states)), batch_size)
        
        states = torch.tensor([self.states[i] for i in index], device = device)
        actions = torch.tensor([self.actions[i] for i in index], device = device)
        rewards = torch.tensor([self.rewards[i] for i in index], device = device)
        next_states = torch.tensor([self.next_states[i] for i in index], device = device)
        dones = torch.tensor([self.dones[i] for i in index], device = device)
        
        return states, actions, rewards, next_states, dones

        #assert NotImplementedError
    

class DQN(nn.Module):
    '''
    Pytorch module for Deep Q Network
    '''
    def __init__(self, input_size, output_size):
        '''
        Define your architecture here
        '''
        super().__init__()
        hidden_size = 256
        #self.example_layer = nn.Linear(input_size, output_size)
        #self.example_activation = nn.Tanh()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,output_size)
        )
        
       

    def forward(self, state):
        '''
        Get Q values for each action given state
        '''
        #q_values = self.example_layer(state)
        q_values = self.net(torch.Tensor(state))
        
        return q_values


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.curr_step = 0

        self.learning_rate = 0.0003
        self.buffer_size = 3000
        self.batch_size = 64
        self.epsilon = 0.10
        self.gamma = 0.85
        self.n_step = 4
        self.target_update_freq = 512
        self.gradient_update_freq = 1
        self.device = torch.device('cpu')
        # self.device = torch.device('cuda')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = DQN(state_size, action_size).to(self.device)
        self.target_network = deepcopy(self.network)

        self.optimizer = torch.optim.Adam(params=self.network.parameters(), lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(buffer_size=self.buffer_size, n_step=self.n_step, gamma=self.gamma)


    def select_action(self, state, is_test=False):
        '''
        selects action given state
        
        returns:
            discrete action integer
        '''
        q = self.network(state)
        rand = np.random.uniform(0,1,1)
        #print(self.network(state))
        if rand < self.epsilon:
            return np.random.randint(0,1)
        else:
            return q.argmax().item()

        #return np.random.randint(self.action_size)


    def train_network(self, states, actions, rewards, next_states, dones):
        # print('ttt')
        # print(len(self.replay_buffer))
        # print(self.buffer_size)
        
        #q_values = q_values.gather(1, actions)
        # q_list = []
        # for i in range(len(q_values)):
        #     q_list.append(q_values[i][actions[i]])
        # realq_values = torch.Tensor(q_list)
        #print('train!!')
        
        if len(self.replay_buffer) < self.buffer_size:
            return
        
        q_values = self.network(states).gather(1,actions.unsqueeze(1)).squeeze(1)
        
        
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values
        
        #loss = F.mse_loss(q_values, target_q_values)
        loss = F.smooth_l1_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.curr_step % self.target_update_freq == 0:
            self.update_target_network()
            self.epsilon *= 0.99
            
        #assert NotImplementedError


    def update_target_network(self):
        '''
        updates the target network to online
        '''
        self.target_network = deepcopy(self.network)
        # Use deepcopy of online network
        #assert NotImplementedError


    def step(self, state, action, reward, next_state, done):
        self.curr_step += 1
        if state[0] > 0.5:
            reward += 1
        self.replay_buffer.add(state, action, reward, next_state, done)

        if len(self.replay_buffer) > self.batch_size and self.curr_step % self.gradient_update_freq == 0:
            self.train_network(*self.replay_buffer.sample(self.batch_size, device=self.device))

