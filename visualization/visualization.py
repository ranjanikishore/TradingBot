from typing import Dict, List
import matplotlib.pyplot as plt
from utils.logging import setup_logging

logger = setup_logging(__name__)

def plot_results(iteration_data: List[Dict], strategy_history: List[Dict]):
    logger.info("Generating visualization")
    steps = [d['step'] for d in iteration_data]
    portfolio_values = [d['portfolio_value'] for d in iteration_data]
    rewards = [d['reward'] for d in iteration_data]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(steps, portfolio_values, label='Portfolio Value')
    plt.xlabel('Step')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Portfolio Value Over Time')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(steps, rewards, label='Reward', color='orange')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward Over Time')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()
    logger.info("Visualization saved to training_results.png")