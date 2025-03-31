Cryptocurrency Trading Bot with Reinforcement Learning

This project implements an automated trading bot for Bitcoin using the **Soft Actor-Critic (SAC)** reinforcement learning algorithm. The bot trades based on historical data, leveraging technical indicators and dynamic strategies to optimize portfolio performance.

Features
- **Reinforcement Learning with SAC**: Utilizes the SAC algorithm with decaying entropy to balance exploration and exploitation, improving trading decisions over time.
- **Adaptive Reward Function**: Implements an adaptive Sharpe ratio with downside penalization to prioritize risk-adjusted returns and penalize large losses.
- **Dynamic Strategy Integration**: Supports trading strategies like MACD Crossover, with a fixed state dimension to ensure consistency.
- **Modular Design**: Code is organized into focused modules (e.g., `domain/`, `application/`) following the Single Responsibility Principle.
- **Efficient Implementation**: Uses `collections.deque` for O(1) replay buffer operations and vectorized computations with NumPy and PyTorch.
- **Testing and Robustness**: Includes unit tests with `pytest` and comprehensive error handling for reliability.

Technologies Used
- **Python**: Core programming language.
- **PyTorch**: For neural network implementation and training.
- **NumPy & Pandas**: For data processing and vectorized operations.
- **pytest**: For unit and integration testing.
- **Gym**: For environment setup following OpenAI Gym conventions.

Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ranjanikishore/trading-bot.git
   cd trading-bot
