import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import os
from typing import List, Tuple, Dict

# Define experience tuple for memory replay buffer
Experience = namedtuple('Experience', 
                        field_names=['state', 'action', 'reward', 'next_state', 'done'])

class QNetwork(nn.Module):
    """Neural network for approximating Q-values"""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [128, 64]):
        """
        Initialize the Q-Network
        
        Args:
            state_size (int): Dimension of state
            action_size (int): Dimension of action space
            hidden_sizes (List[int]): List of hidden layer sizes
        """
        super(QNetwork, self).__init__()
        
        # Input layer
        layers = [nn.Linear(state_size, hidden_sizes[0]), nn.ReLU()]
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], action_size))
        
        # Combine all layers
        self.model = nn.Sequential(*layers)
        
    def forward(self, state):
        """
        Forward pass through the network
        
        Args:
            state: State tensor
            
        Returns:
            Q-values for each action
        """
        return self.model(state)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples"""
    
    def __init__(self, buffer_size: int, batch_size: int, device: torch.device):
        """
        Initialize the replay buffer
        
        Args:
            buffer_size (int): Maximum size of buffer
            batch_size (int): Size of each training batch
            device (torch.device): Device to store the tensors
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=min(self.batch_size, len(self.memory)))
        
        # Convert to tensors and move to the appropriate device
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of the buffer"""
        return len(self.memory)


class DQNAgent:
    """Agent implementing Deep Q-Learning for book recommendations"""
    
    def __init__(self, 
                 state_size: int, 
                 action_size: int, 
                 learning_rate: float = 5e-4,
                 hidden_sizes: List[int] = [128, 64],
                 buffer_size: int = 10000,
                 batch_size: int = 64, 
                 gamma: float = 0.99, 
                 tau: float = 1e-3, 
                 update_every: int = 4,
                 device: torch.device = None):
        """
        Initialize a DQN Agent
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            learning_rate (float): Learning rate for the optimizer
            hidden_sizes (List[int]): Sizes of hidden layers in the Q-Network
            buffer_size (int): Size of the replay buffer
            batch_size (int): Size of the training batch
            gamma (float): Discount factor
            tau (float): Soft update parameter for target network
            update_every (int): How often to update the network
            device (torch.device): Device to run the model on
        """
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize parameters
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.learning_rate = learning_rate
        
        # Q-Networks (online and target)
        self.qnetwork_local = QNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_sizes).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, self.device)
        
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """
        Update agent's knowledge based on experience
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            # Get random batch of experiences
            experiences = self.memory.sample()
            # Update Q-values
            self._learn(experiences)
    
    def act(self, state, eps: float = 0.0):
        """
        Returns actions for given state as per current policy
        
        Args:
            state: Current state
            eps (float): Epsilon for epsilon-greedy action selection
            
        Returns:
            int: Selected action
        """
        # Convert state to tensor and move to device
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Set eval mode (no gradient tracking)
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            
        # Set back to train mode
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return random.choice(np.arange(self.action_size))
    
    def _learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples
        
        Args:
            experiences: Tuple of (states, actions, rewards, next_states, dones)
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target)
    
    def _soft_update(self, local_model, target_model):
        """
        Soft update model parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            local_model: Source model
            target_model: Target model to update
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, filepath: str):
        """
        Save the model parameters
        
        Args:
            filepath (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state
        torch.save({
            'qnetwork_local_state_dict': self.qnetwork_local.state_dict(),
            'qnetwork_target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load the model parameters
        
        Args:
            filepath (str): Path to load the model from
        """
        if not os.path.exists(filepath):
            print(f"No model found at {filepath}")
            return
            
        # Load model state
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local_state_dict'])
        self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from {filepath}")
    
    def get_average_q_values(self, state):
        """
        Calculate the average Q-values for a given state
        
        Args:
            state: The state to calculate Q-values for
            
        Returns:
            float: Average of Q-values across all actions
        """
        # Convert state to tensor and move to device
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Set eval mode (no gradient tracking)
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            q_values = self.qnetwork_local(state_tensor)
            
        # Set back to train mode
        self.qnetwork_local.train()
        
        # Return average Q-value
        return q_values.mean().item() 