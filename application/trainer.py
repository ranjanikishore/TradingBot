from typing import Dict, List
import numpy as np
from config import Config
from domain.environment import CryptoTradingEnv
from domain.agent import SACAgent
from utils.logging import setup_logging

logger = setup_logging(__name__)

class Trainer:
    def __init__(self, env: CryptoTradingEnv, agent: SACAgent):
        self.__env = env
        self.__agent = agent
        self.__max_steps = Config.MAX_STEPS
        self.__patience = Config.PATIENCE_EARLY_STOPPING
        self.__current_state_dim = env.observation_space.shape[0]
        logger.debug(f"Trainer initialized with max_steps={self.__max_steps}, initial state_dim={self.__current_state_dim}")
    
    def __check_and_update_agent(self, new_state_dim: int) -> None:
        if new_state_dim != self.__current_state_dim:
            logger.info(f"Observation space changed from {self.__current_state_dim} to {new_state_dim}. Updating agent state dimension.")
            self.__agent.update_state_dim(new_state_dim)
            self.__current_state_dim = new_state_dim
    
    def train(self) -> List[Dict]:
        logger.info("Starting training loop")
        state = self.__env.reset()
        logger.debug(f"Initial state shape: {state.shape}")
        iteration_data = []
        best_value = -float('inf')
        patience_counter = 0
        
        for step in range(self.__max_steps):
            logger.debug(f"Step {step + 1}: Selecting action")
            self.__check_and_update_agent(self.__env.observation_space.shape[0])
            action = self.__agent.select_action(state)
            logger.debug(f"Step {step + 1}: Action selected: {action}")
            next_state, reward, done, info = self.__env.step(action)
            logger.debug(f"Step {step + 1}: Environment stepped, reward={reward}, done={done}")
            self.__agent.replay_buffer.push(state, action, reward, next_state, done)
            logger.debug(f"Step {step + 1}: Experience added to replay buffer")
            self.__agent.train()
            logger.debug(f"Step {step + 1}: Agent trained")
            
            current_data = self.__env.data.iloc[self.__env.current_step - 1]
            strategy_signals = self.__env._CryptoTradingEnv__strategy_manager.evaluate_strategies(self.__env.data, self.__env.current_step - 1)
            return_value = (info['value'] - (self.__env.balance + self.__env.btc_held * current_data['close'])) / (self.__env.balance + 1e-6)
            for name, signal in strategy_signals.items():
                if signal == np.sign(action):
                    success = return_value > 0
                    self.__agent.remember_strategy(name, success, return_value, current_data)
            
            iteration_info = {
                'step': step + 1,
                'state': state.tolist(),
                'action': float(action),
                'reward': float(reward),
                'next_state': next_state.tolist(),
                'done': done,
                'portfolio_value': info['value'],
                'volatility': info['volatility'],
                'btc_held': self.__env.btc_held,
                'balance': self.__env.balance,
                'timestamp': str(self.__env.data.iloc[self.__env.current_step]['timestamp'])
            }
            iteration_data.append(iteration_info)
            
            current_value = info['value']
            if current_value > best_value:
                best_value = current_value
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.__patience:
                logger.info(f"Early stopping at step {step + 1}: Portfolio value not improving")
                break
            
            state = next_state
            if done:
                state = self.__env.reset()
                logger.debug(f"Environment reset at step {step + 1}")
        
        logger.info("Training loop completed")
        return iteration_data