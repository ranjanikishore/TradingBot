from typing import Dict, List
import pandas as pd
from utils.logging import setup_logging

logger = setup_logging(__name__)

class TradingStrategy:
    def __init__(self, name: str):
        self.name = name
        self.__returns: List[float] = []
        self.__pitfalls: List[Dict] = []
    
    def evaluate(self, data: pd.DataFrame, index: int) -> float:
        raise NotImplementedError("Subclasses must implement evaluate")
    
    def update_performance(self, return_value: float):
        self.__returns.append(return_value)
    
    def log_pitfall(self, return_value: float, data: pd.Series):
        if return_value < 0:
            self.__pitfalls.append({"return": return_value, "data": data.to_dict()})
            logger.debug(f"Pitfall logged for {self.name}: Return={return_value}")
    
    def get_success_rate(self) -> float:
        if not self.__returns:
            return 0.0
        return sum(1 for r in self.__returns if r > 0) / len(self.__returns)

class MACDCrossover(TradingStrategy):
    def evaluate(self, data: pd.DataFrame, index: int) -> float:
        if index < 1:  # Need at least one previous row
            return 0.0
        current_macd = data['macd'].iloc[index]
        current_signal = data['macd_signal'].iloc[index]
        prev_macd = data['macd'].iloc[index - 1]
        prev_signal = data['macd_signal'].iloc[index - 1]
        if current_macd > current_signal and prev_macd <= prev_signal:
            return 1.0  # Buy
        elif current_macd < current_signal and prev_macd >= prev_signal:
            return -1.0  # Sell
        return 0.0

class RSIOverboughtOversold(TradingStrategy):
    def evaluate(self, data: pd.DataFrame, index: int) -> float:
        rsi = data['rsi_14'].iloc[index]
        if rsi > 70:
            return -1.0  # Sell
        elif rsi < 30:
            return 1.0  # Buy
        return 0.0

class IchimokuCloud(TradingStrategy):
    def evaluate(self, data: pd.DataFrame, index: int) -> float:
        close = data['close'].iloc[index]
        ichimoku_a = data['ichimoku_a'].iloc[index]
        ichimoku_b = data['ichimoku_b'].iloc[index]
        if close > ichimoku_a and close > ichimoku_b:
            return 1.0  # Buy
        elif close < ichimoku_a and close < ichimoku_b:
            return -1.0  # Sell
        return 0.0

class ADXTrend(TradingStrategy):
    def evaluate(self, data: pd.DataFrame, index: int) -> float:
        adx = data['adx'].iloc[index]
        adx_pos = data['adx_pos'].iloc[index]
        adx_neg = data['adx_neg'].iloc[index]
        recent_vol = data['atr_14'].iloc[max(0, index-20):index+1].mean()  # 20-period ATR
        adx_threshold = 25 * (1 + recent_vol / data['atr_14'].mean())  # Dynamic threshold
        if adx > adx_threshold and adx_pos > adx_neg:
            return 1.0
        elif adx > adx_threshold and adx_neg > adx_pos:
            return -1.0
        return 0.0

class StrategyManager:
    def __init__(self):
        self.__strategies: List[TradingStrategy] = []
        self.history: List[Dict] = []
    
    def add_strategy(self, strategy: TradingStrategy):
        self.__strategies.append(strategy)
        logger.debug(f"Added strategy: {strategy.name}")
    
    def evaluate_strategies(self, data: pd.DataFrame, index: int) -> Dict[str, float]:
        signals = {strategy.name: strategy.evaluate(data, index) for strategy in self.__strategies}
        current_row = data.iloc[index]
        # Convert Timestamp to string for JSON compatibility
        self.history.append({"timestamp": current_row['timestamp'].isoformat(), "signals": signals})
        return signals
    
    def update_performance(self, strategy_name: str, return_value: float):
        for strategy in self.__strategies:
            if strategy.name == strategy_name:
                strategy.update_performance(return_value)
                break
    
    def log_pitfalls(self, strategy_name: str, return_value: float, data: pd.Series):
        for strategy in self.__strategies:
            if strategy.name == strategy_name:
                strategy.log_pitfall(return_value, data)
                break
    
    def get_successful_strategies(self, threshold: float) -> List[TradingStrategy]:
        return [s for s in self.__strategies if s.get_success_rate() > threshold]

def create_initial_strategies(manager: StrategyManager):
    manager.add_strategy(MACDCrossover("macd_crossover"))
    manager.add_strategy(RSIOverboughtOversold("rsi_overbought"))
    manager.add_strategy(IchimokuCloud("ichimoku_cloud"))
    manager.add_strategy(ADXTrend("adx_trend"))
    logger.info("Initial strategies created")