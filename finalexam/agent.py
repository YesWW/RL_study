import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import pickle

def get_demo_traj():
    with open('demonstration.pkl', 'rb') as fr:
        demonstration = pickle.load(fr)

    return demonstration


class DQfDNetwork(nn.Module):
    '''
    Pytorch module for Deep Q Network
    '''
    def __init__(self, state_dim, action_dim, hidden_size):
        '''
        Define your Q network architecture here
        '''
        # TODO
        super(DQfDNetwork, self).__init__() 
        
        self.qnet = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,action_dim)
        )
        #assert NotImplementedError

    def forward(self, state):
        '''
        Get Q values for actions given state
        '''
        # TODO
        action = self.qnet(state)
        return action
        #assert NotImplementedError
    

class ReplayMemory(object):
    '''
    Save transition datas to buffer
    '''
    def __init__(self, n_step, gamma, buffer_size=50000):
        # TODO
        self.nstep = n_step
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.buffer = []
        #assert NotImplementedError

    def add(self, transition, is_demo=False):
        '''
        Add samples to the buffer
        '''
        # TODO
        
        self.buffer.append(transition)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
        #assert NotImplementedError

    def sample(self, batch_size):
        '''
        samples random batches from buffer
        '''
        # TODO
        index = np.random.choice(len(self.buffer), batch_size)
        sample = [self.buffer[i] for i in index]
        return sample
        #assert NotImplementedError


class DQfDAgent(object):
    '''
    DQfD train agent
    '''
    def __init__(self, env, state_dim, action_dim):
        # DQN hyperparameters
        self.lr = 0.0005
        self.gamma = 0.99
        self.epsilon = 0.01
        self.target_update_freq = 250  #10000
        self.hidden_size = 128

        # DQfD loss function hyperparameters (Check the DQfD paper published in AAAI 2018)
        self.n_step = 10
        self.margin = 0.8
        self.lambda_1 = 1.0
        self.lambda_2 = 1.0
        self.lambda_3 = 1e-5

        self.env = env
        self.main_net = DQfDNetwork(state_dim, action_dim, self.hidden_size)
        self.target_net = DQfDNetwork(state_dim, action_dim, self.hidden_size)
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=self.lr)
        self.memory = ReplayMemory(self.n_step, self.gamma)

    def get_action(self, state):
        '''
        Select actions for given states with epsilon greedy
        '''
        # TODO
        if np.random.rand() < self.epsilon:
            action = torch.tensor(np.random.randint(2))
        else:
            q_values = self.main_net(state)
            action = torch.argmax(q_values, dim=0)
            #print(action)
        return action
        #assert NotImplementedError

    def calculate_loss(self, mini_batch):
        '''
        Implement DQfD loss function
        '''
        states = torch.tensor([data[0] for data in mini_batch], dtype=torch.float32)
        actions = torch.tensor([data[1] for data in mini_batch], dtype=torch.int64)
        rewards = torch.tensor([data[2] for data in mini_batch], dtype=torch.float32)
        next_states = torch.tensor([data[3] for data in mini_batch], dtype=torch.float32)
        dones = torch.tensor([data[4] for data in mini_batch], dtype=torch.float32)
        is_demo = torch.tensor([data[5] for data in mini_batch], dtype=torch.float32)
        
        # one-step double Q-learning loss calculation
        # TODO
        q_ = self.main_net(states).gather(1, actions.unsqueeze(1)).unsqueeze(1)
        with torch.no_grad():
            next_Q = self.target_net(next_states)
            max_next_Q = torch.max(next_Q, dim=1)[0]
            target_Q = rewards + self.gamma * max_next_Q * (1 - dones)
        one_step_loss = self.loss_func(q_, target_Q)
        
        # n-step double Q-learning loss calculation
        # TODO
        n_loss = torch.tensor(0.0)
        if len(mini_batch) >= self.n_step:
            n_states = torch.tensor([data[0] for data in mini_batch[-self.n_step:]], dtype=torch.float32)
            n_actions = torch.tensor([data[1] for data in mini_batch[-self.n_step:]], dtype=torch.int64)
            n_rewards = torch.tensor([data[2] for data in mini_batch[-self.n_step:]], dtype=torch.float32)
            n_next_states = torch.tensor([data[3] for data in mini_batch[-self.n_step:]], dtype=torch.float32)
            n_dones = torch.tensor([data[4] for data in mini_batch[-self.n_step:]], dtype=torch.float32)
            
            n_q_ = self.main_net(n_states).gather(1, n_actions.unsqueeze(1)).unsqueeze(1)
            with torch.no_grad():
                n_next_Q = self.target_net(n_next_states)
                n_max_next_Q = torch.max(n_next_Q, dim=1)[0]
                n_target_Q = n_rewards + self.gamma**self.n_step * n_max_next_Q * (1 - n_dones)
            n_loss = self.loss_func(n_q_, n_target_Q)
        
        # supervised large margin classification loss calculation
        # TODO        
        # state = states[0]
        # action = actions[0]
        # q = self.main_net(state)
        # q_max, index = torch.max(q, dim=0)
        # diff = abs(index - action)
        # Qe = q[action]
        # if is_demo.numpy().all():
        #     supervised_loss = q_max + diff - Qe
        # else:
        #     supervised_loss = torch.tensor(0.0)
            
        sQ = self.main_net(states)
        q_max, index = torch.max(sQ, dim=1)
        diff = torch.abs(index - actions)
        Qe = sQ[torch.arange(sQ.size(0)), actions]

        if is_demo.numpy().all():
            supervised_loss = torch.max(q_max + diff - Qe, torch.tensor(0.0))[0]
        else:
            supervised_loss = torch.tensor(0.0)
        

        # L2 regularization loss calculation
        # TODO
        l2_loss = torch.tensor(0.0)
        for param in self.main_net.parameters():
            l2_loss += torch.norm(param, p=2)

        # Total loss calculation
        # TODO
        total_loss = (
            one_step_loss +
            self.lambda_1 * n_loss +
            self.lambda_2 * supervised_loss +
            self.lambda_3 * l2_loss
        )
        # print('one :', one_step_loss)
        # print('n-step :', n_loss)
        # print('super :', supervised_loss)
        # print('l2 :', l2_loss)
        
        return total_loss
        #return one_step_loss
        #assert NotImplementedError

    def pretrain(self):
        '''
        DQfD pre-train with the demonstration dataset
        '''
        demonstration = get_demo_traj()
        #print(demonstration)
        # Add the demonstration dataset into the replay buffer
        # TODO
        data = []
        for j in range(len(demonstration['states'])):
            for i in demonstration:
                data.append(demonstration[i][j])
            data.append(True)
            self.memory.add(data)
            data = []
        # Pre-train for 1000 steps
        for pretrain_step in range(1000):
            pretrain_batch = self.memory.sample(batch_size=32)
            loss = self.calculate_loss(pretrain_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if pretrain_step % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.main_net.state_dict())

    def train(self):
        '''
        DQfD main train function
        '''
        ################################  DO NOT CHANGE  ################################
        episode_scores = deque(maxlen=20)
        mean_scores = []
        train_step = 0
        
        # Pre-train with the demonstration data 
        self.pretrain()       

        for episode in range(250):
            score = 0            
            done = False
            state = self.env.reset()

            while not done:
                action = self.get_action(torch.FloatTensor(state)).item()
                next_state, reward, done, _ = self.env.step(action)
                score += reward      
        #################################################################################

                # Train while interacting with the environment after pre-train
                # TODO
                data = (state, action, reward, next_state, done, False)
                self.memory.add(data)
                if train_step > 1000:
                    mini_batch = self.memory.sample(batch_size=32)
                    loss = self.calculate_loss(mini_batch)
                    #print(loss)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                if train_step % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.main_net.state_dict())
                
        ################################  DO NOT CHANGE  ################################
                train_step += 1

                if done:
                    episode_scores.append(score)
                    mean_score = np.mean(episode_scores)
                    mean_scores.append(mean_score)
                    print(f'[Episode {episode}] Avg. score: {mean_score}')

            if (mean_score > 475) and (len(episode_scores) == 20):
                break

        return mean_scores
        #################################################################################
