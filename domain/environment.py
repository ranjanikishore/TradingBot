import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from config import Config
from domain.strategies import StrategyManager
from utils.logging import setup_logging

logger = setup_logging(__name__)

class CryptoTradingEnv(gym.Env):
    DATAFRAME_FEATURES = [
        'close', 'volume', 'rsi_14', 'rsi_7', 'stochastic_k', 'stochastic_d', 'williams_r', 'roc_5', 'roc_10',
        'ema_10', 'ema_20', 'sma_50', 'sma_200', 'macd', 'macd_signal', 'adx', 'adx_pos', 'adx_neg',
        'bollinger_mavg', 'bollinger_hband', 'bollinger_lband', 'bollinger_width', 'atr_14', 'atr_7',
        'obv', 'cmf', 'vwap', 'ichimoku_a', 'ichimoku_b', 'ichimoku_base', 'ichimoku_conversion', 'cci', 'psar',
        'return_1', 'return_5', 'return_20'
    ]
    NUM_DATAFRAME_FEATURES = len(DATAFRAME_FEATURES)
    NUM_ADDITIONAL_FEATURES = 14

    def __init__(self, data: pd.DataFrame, initial_balance: float = Config.INITIAL_BALANCE):
        super().__init__()
        self.__data = data
        self.__current_step = 0
        self.__initial_balance = initial_balance
        self.__balance = initial_balance
        self.__btc_held = 0.0
        self.__commission = Config.COMMISSION
        self.__max_position = Config.MAX_POSITION
        self.__min_hold_steps = Config.MIN_HOLD_STEPS
        self.__steps_since_trade = 0
        self.__last_trade_step = -self.__min_hold_steps
        
        self.__price_history = []
        self.__max_history = Config.MAX_HISTORY
        self.__strategy_manager = StrategyManager()
        self.__strategy_features = []
        
        from domain.strategies import create_initial_strategies
        create_initial_strategies(self.__strategy_manager)
        
        sample_observation = self.__next_observation()
        self.__current_step = 0
        observation_shape = sample_observation.shape[0]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_shape,), dtype=np.float32)
        
        logger.info(f"Environment initialized with {len(self.__data)} data points, observation shape: {observation_shape}")

    @property
    def btc_held(self) -> float:
        return self.__btc_held
    
    @property
    def balance(self) -> float:
        return self.__balance
    
    @property
    def current_step(self) -> int:
        return self.__current_step
    
    @property
    def data(self) -> pd.DataFrame:
        return self.__data.copy()
    
    def __get_volatility(self) -> float:
        if len(self.__price_history) < 2:
            return 0.0
        returns = np.diff(np.log(self.__price_history))
        return np.std(returns) * np.sqrt(252)
    
    def __get_dataframe_features(self) -> np.ndarray:
        features = self.__data.iloc[self.__current_step][self.DATAFRAME_FEATURES].values.astype(np.float32)
        if len(features) != self.NUM_DATAFRAME_FEATURES:
            raise ValueError(
                f"DataFrame features mismatch: expected {self.NUM_DATAFRAME_FEATURES}, got {len(features)}"
            )
        return features
    
    def __get_additional_features(self, current_price: float, portfolio_value: float) -> list:
        additional = [
            (self.__current_step % 24) / 24,
            self.__balance / self.__initial_balance,
            self.__btc_held * current_price / self.__initial_balance,
            self.__get_volatility(),
            portfolio_value / self.__initial_balance,
            current_price / self.__data['close'].mean(),
            self.__data['volume'].iloc[self.__current_step] / self.__data['volume'].mean(),
            self.__btc_held,
            self.__balance / current_price if current_price != 0 else 0.0,
            float(self.__current_step / len(self.__data)),
            float(np.log1p(self.__current_step)),
            self.__data['rsi_14'].iloc[self.__current_step] / 100.0,
            self.__data['stochastic_k'].iloc[self.__current_step] / 100.0,
            (self.__data['close'].iloc[self.__current_step] - self.__data['sma_50'].iloc[self.__current_step]) / self.__data['sma_50'].iloc[self.__current_step]
        ]
        if len(additional) != self.NUM_ADDITIONAL_FEATURES:
            raise ValueError(
                f"Additional features mismatch: expected {self.NUM_ADDITIONAL_FEATURES}, got {len(additional)}"
            )
        return additional
    
    def __get_strategy_features(self) -> np.ndarray:
        strategy_signals = self.__strategy_manager.evaluate_strategies(self.__data, self.__current_step)
        return np.array([strategy_signals.get(feat, 0.0) for feat in self.__strategy_features], dtype=np.float32)
    
    def __next_observation(self) -> np.ndarray:
        current_price = self.__data.iloc[self.__current_step]['close']
        self.__price_history.append(current_price)
        if len(self.__price_history) > self.__max_history:
            self.__price_history.pop(0)
        portfolio_value = self.__balance + self.__btc_held * current_price
        
        dataframe_features = self.__get_dataframe_features()
        additional_features = self.__get_additional_features(current_price, portfolio_value)
        strategy_features = self.__get_strategy_features()
        
        state = np.concatenate([dataframe_features, additional_features, strategy_features])
        logger.debug(f"Observation shape at step {self.__current_step}: {state.shape}")
        return state
    
    def __update_observation_space(self):
        successful_strategies = self.__strategy_manager.get_successful_strategies(Config.STRATEGY_SUCCESS_THRESHOLD)
        new_strategy_features = [s.name for s in successful_strategies if s.name not in self.__strategy_features]
        if new_strategy_features and len(self.__strategy_features) < Config.MAX_STRATEGIES:
            self.__strategy_features.extend(new_strategy_features[:Config.MAX_STRATEGIES - len(self.__strategy_features)])
            new_shape = self.NUM_DATAFRAME_FEATURES + self.NUM_ADDITIONAL_FEATURES + len(self.__strategy_features)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(new_shape,), dtype=np.float32)
            logger.info(f"Updated observation space to include {len(self.__strategy_features)} strategies: {new_shape}")
    
    def step(self, action: float) -> tuple:
        logger.debug(f"Stepping environment with action {action}")
        action = np.clip(action, -1, 1)
        current_price = self.__data.iloc[self.__current_step]['close']
        portfolio_value = self.__balance + self.__btc_held * current_price
        max_trade_value = portfolio_value * self.__max_position
        
        self.__steps_since_trade = self.__current_step - self.__last_trade_step
        trade_allowed = self.__steps_since_trade >= self.__min_hold_steps
        
        delta_btc = 0.0
        trade_value = 0.0
        trade_cost = 0.0
        
        if trade_allowed:
            target_btc = np.clip((action * portfolio_value) / current_price, 0, max_trade_value / current_price)
            delta_btc = target_btc - self.__btc_held
            trade_value = delta_btc * current_price
            trade_cost = abs(trade_value) * self.__commission
            self.__balance -= trade_value + trade_cost
            self.__btc_held = target_btc
        
        if abs(delta_btc) > 1e-6:
            self.__last_trade_step = self.__current_step
        
        self.__current_step += 1
        done = self.__current_step == len(self.__data) - 1
        new_price = self.__data.iloc[self.__current_step]['close']
        new_value = self.__balance + self.__btc_held * new_price
        
        stop_loss = max(0.02, min(0.1, self.__get_volatility() * 0.5))
        price_change = (new_price - current_price) / current_price
        stop_loss_triggered = False
        if self.__btc_held > 0 and abs(price_change) >= stop_loss:
            trade_cost += abs(self.__btc_held * new_price) * self.__commission
            self.__balance += self.__btc_held * new_price * (1 - self.__commission)
            self.__btc_held = 0
            new_value = self.__balance
            stop_loss_triggered = True
            logger.info(f"Stop-loss triggered at step {self.__current_step}: {price_change*100:.2f}% change")
        
        holding_reward = Config.HOLDING_REWARD_COEFFICIENT * price_change if self.__btc_held > 0 and price_change > 0 else 0
        reward = (
            np.log(new_value / (portfolio_value + 1e-6)) +
            holding_reward -
            Config.TRANSACTION_COST_PENALTY * trade_cost / (portfolio_value + 1e-6)
        )
        
        current_data = self.__data.iloc[self.__current_step - 1]
        strategy_signals = self.__strategy_manager.evaluate_strategies(self.__data, self.__current_step - 1)
        return_value = price_change if self.__btc_held > 0 else 0.0
        for name, signal in strategy_signals.items():
            if signal == np.sign(action):
                self.__strategy_manager.update_performance(name, return_value)
                self.__strategy_manager.log_pitfalls(name, return_value, current_data)
        
        self.__update_observation_space()
        
        logger.info(f"Step {self.__current_step}: Value=${new_value:,.2f}, Action={action:.2f}, Reward={reward:.4f}, "
                    f"Price Change={price_change*100:.2f}%, BTC Held={self.__btc_held:.6f}, "
                    f"Balance=${self.__balance:,.2f}, Trade Cost=${trade_cost:.2f}, Stop-Loss={'Yes' if stop_loss_triggered else 'No'}")
        return self.__next_observation(), reward, done, {'value': new_value, 'volatility': self.__get_volatility()}
    
    def reset(self) -> np.ndarray:
        logger.debug("Resetting environment")
        self.__current_step = 0
        self.__balance = self.__initial_balance
        self.__btc_held = 0.0
        self.__price_history = []
        self.__last_trade_step = -self.__min_hold_steps
        logger.info(f"Environment reset: Balance=${self.__balance:,.2f}, BTC Held={self.__btc_held:.6f}")
        state = self.__next_observation()
        logger.debug(f"Reset state shape: {state.shape}")
        return state