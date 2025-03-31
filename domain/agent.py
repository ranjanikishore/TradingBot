import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from config import Config
from domain.models import Actor, Critic
from utils.logging import setup_logging

logger = setup_logging(__name__)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.__buffer = []
        self.__capacity = capacity

    def push(self, state: np.ndarray, action: float, reward: float, next_state: np.ndarray, done: bool) -> None:
        if len(self.__buffer) >= self.__capacity:
            self.__buffer.pop(0)
        self.__buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, current_state_dim: int) -> tuple:
        indices = np.random.randint(0, len(self.__buffer), size=batch_size)
        states, actions, rewards, next_states, dones = zip(*[self.__buffer[i] for i in indices])

        # Pad states and next_states to match current_state_dim
        def pad_to_dim(arr_list, target_dim):
            padded = []
            for arr in arr_list:
                if arr.shape[0] < target_dim:
                    pad_width = target_dim - arr.shape[0]
                    padded.append(np.pad(arr, (0, pad_width), mode='constant', constant_values=0))
                else:
                    padded.append(arr)
            return np.stack(padded)  # Stack into a 2D array (e.g., (64, 51))

        return (
            torch.FloatTensor(pad_to_dim(states, current_state_dim)),
            torch.FloatTensor(actions).reshape(-1, 1),
            torch.FloatTensor(rewards).reshape(-1, 1),
            torch.FloatTensor(pad_to_dim(next_states, current_state_dim)),
            torch.FloatTensor(dones).reshape(-1, 1)
        )

    def __len__(self) -> int:
        return len(self.__buffer)

class SACAgent:
    def __init__(self, state_dim: int, action_dim: int):
        logger.debug(f"Initializing SACAgent with state_dim={state_dim}, action_dim={action_dim}")
        self.__state_dim = state_dim
        self.__action_dim = action_dim
        self.__actor = Actor(state_dim, action_dim)
        self.__critic1 = Critic(state_dim, action_dim)
        self.__critic2 = Critic(state_dim, action_dim)
        self.__target_critic1 = Critic(state_dim, action_dim)
        self.__target_critic2 = Critic(state_dim, action_dim)
        self.__hard_update(self.__target_critic1, self.__critic1)
        self.__hard_update(self.__target_critic2, self.__critic2)
        
        self.__actor_optim = optim.Adam(self.__actor.parameters(), lr=Config.LEARNING_RATE)
        self.__critic1_optim = optim.Adam(self.__critic1.parameters(), lr=Config.LEARNING_RATE)
        self.__critic2_optim = optim.Adam(self.__critic2.parameters(), lr=Config.LEARNING_RATE)
        
        self.__actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.__actor_optim, patience=Config.PATIENCE, factor=Config.LR_FACTOR)
        self.__critic1_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.__critic1_optim, patience=Config.PATIENCE, factor=Config.LR_FACTOR)
        self.__critic2_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.__critic2_optim, patience=Config.PATIENCE, factor=Config.LR_FACTOR)
        
        self.__gamma = Config.GAMMA
        self.__tau = Config.TAU
        self.__batch_size = Config.BATCH_SIZE
        self.__replay_buffer = ReplayBuffer(Config.REPLAY_BUFFER_CAPACITY)
        
        self.__log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.__alpha_optim = optim.Adam([self.__log_alpha], lr=Config.LEARNING_RATE)
        self.__target_entropy = -action_dim * Config.ENTROPY_TARGET_MULTIPLIER
        
        self.__strategy_memory = []
        
        logger.info(f"SAC Agent initialized with state_dim={state_dim}")

    @property
    def replay_buffer(self) -> ReplayBuffer:
        return self.__replay_buffer
    
    def __hard_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def __soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.__tau * param.data + (1 - self.__tau) * target_param.data)
    
    def update_state_dim(self, new_state_dim: int) -> None:
        if new_state_dim <= self.__state_dim:
            logger.debug(f"New state_dim {new_state_dim} <= current {self.__state_dim}, no update needed")
            return
        
        logger.info(f"Updating agent state dimension from {self.__state_dim} to {new_state_dim}")
        
        # Create new networks with updated state_dim
        new_actor = Actor(new_state_dim, self.__action_dim)
        new_critic1 = Critic(new_state_dim, self.__action_dim)
        new_critic2 = Critic(new_state_dim, self.__action_dim)
        new_target_critic1 = Critic(new_state_dim, self.__action_dim)
        new_target_critic2 = Critic(new_state_dim, self.__action_dim)
        
        # Copy weights from old networks to new ones
        with torch.no_grad():
            for old_param, new_param in zip(self.__actor.parameters(), new_actor.parameters()):
                if old_param.shape == new_param.shape:
                    new_param.copy_(old_param)
                elif 'linear1.weight' in str(old_param):  # Adjust input layer weights
                    new_param[:, :self.__state_dim].copy_(old_param)
            
            for old_param, new_param in zip(self.__critic1.parameters(), new_critic1.parameters()):
                if old_param.shape == new_param.shape:
                    new_param.copy_(old_param)
                elif 'linear1.weight' in str(old_param):
                    new_param[:, :self.__state_dim].copy_(old_param)
            
            for old_param, new_param in zip(self.__critic2.parameters(), new_critic2.parameters()):
                if old_param.shape == new_param.shape:
                    new_param.copy_(old_param)
                elif 'linear1.weight' in str(old_param):
                    new_param[:, :self.__state_dim].copy_(old_param)
        
        self.__hard_update(new_target_critic1, new_critic1)
        self.__hard_update(new_target_critic2, new_critic2)
        
        # Update optimizers with new parameters
        self.__actor = new_actor
        self.__critic1 = new_critic1
        self.__critic2 = new_critic2
        self.__target_critic1 = new_target_critic1
        self.__target_critic2 = new_target_critic2
        
        self.__actor_optim = optim.Adam(self.__actor.parameters(), lr=Config.LEARNING_RATE)
        self.__critic1_optim = optim.Adam(self.__critic1.parameters(), lr=Config.LEARNING_RATE)
        self.__critic2_optim = optim.Adam(self.__critic2.parameters(), lr=Config.LEARNING_RATE)
        
        self.__actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.__actor_optim, patience=Config.PATIENCE, factor=Config.LR_FACTOR)
        self.__critic1_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.__critic1_optim, patience=Config.PATIENCE, factor=Config.LR_FACTOR)
        self.__critic2_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.__critic2_optim, patience=Config.PATIENCE, factor=Config.LR_FACTOR)
        
        self.__state_dim = new_state_dim
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> float:
        logger.debug(f"Selecting action for state with shape {state.shape}")
        if state.shape[0] < self.__state_dim:
            state = np.pad(state, (0, self.__state_dim - state.shape[0]), mode='constant', constant_values=0)
        state = torch.FloatTensor(state).unsqueeze(0)
        logger.debug(f"State converted to tensor with shape {state.shape}")
        with torch.no_grad():
            dist = self.__actor(state)
            logger.debug("Actor forward pass completed")
            action = dist.mean if deterministic else dist.sample()
            logger.debug(f"Action sampled: {action}")
        return action[0].item()
    
    def train(self) -> None:
        logger.debug("Starting training step")
        buffer_size = len(self.__replay_buffer)
        logger.debug(f"Replay buffer size: {buffer_size}")
        if buffer_size < self.__batch_size:
            logger.debug("Replay buffer too small, skipping training")
            return
        logger.debug("Sampling from replay buffer")
        states, actions, rewards, next_states, dones = self.__replay_buffer.sample(self.__batch_size, self.__state_dim)
        logger.debug(f"Sampled batch with states shape {states.shape}")
        
        alpha = self.__log_alpha.exp()
        
        with torch.no_grad():
            next_dist = self.__actor(next_states)
            next_actions = next_dist.sample()
            next_log_probs = next_dist.log_prob(next_actions).view(-1, 1)
            next_q1 = self.__target_critic1(next_states, next_actions)
            next_q2 = self.__target_critic2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.__gamma * next_q
        
        current_q1 = self.__critic1(states, actions)
        current_q2 = self.__critic2(states, actions)
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)
        
        self.__critic1_optim.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.__critic1.parameters(), max_norm=1.0)
        self.__critic1_optim.step()
        self.__critic1_scheduler.step(critic1_loss)
        
        self.__critic2_optim.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.__critic2.parameters(), max_norm=1.0)
        self.__critic2_optim.step()
        self.__critic2_scheduler.step(critic2_loss)
        
        dist = self.__actor(states)
        new_actions = dist.sample()
        log_probs = dist.log_prob(new_actions).view(-1, 1)
        q_values = self.__critic1(states, new_actions)
        actor_loss = (alpha * log_probs - q_values).mean()
        
        self.__actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.__actor.parameters(), max_norm=1.0)
        self.__actor_optim.step()
        self.__actor_scheduler.step(actor_loss)
        
        alpha_loss = -(self.__log_alpha * (log_probs + self.__target_entropy).detach()).mean()
        self.__alpha_optim.zero_grad()
        alpha_loss.backward()
        self.__alpha_optim.step()
        
        self.__soft_update(self.__target_critic1, self.__critic1)
        self.__soft_update(self.__target_critic2, self.__critic2)
        
        logger.info(f"Training step: Critic1 Loss={critic1_loss.item():.4f}, Critic2 Loss={critic2_loss.item():.4f}, "
                    f"Actor Loss={actor_loss.item():.4f}, Alpha={alpha.item():.4f}")
        logger.debug("Training step completed")
    
    def remember_strategy(self, strategy_name: str, success: bool, return_value: float, data: pd.Series):
        self.__strategy_memory.append({
            "strategy": strategy_name,
            "success": success,
            "return": return_value,
            "data": data.to_dict(),
            "timestamp": pd.Timestamp.now().isoformat()
        })
        logger.info(f"Remembered strategy {strategy_name}: Success={success}, Return={return_value}")