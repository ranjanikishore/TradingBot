import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.trend import PSARIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, VolumeWeightedAveragePrice
from utils.logging import setup_logging

#logger = setup_logging(__name__)

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    #logger.info("Calculating technical indicators")
    
    # Start with a clean copy of the input data
    data = data.copy()
    
    # Add basic TA features (covers some indicators automatically)
    data = add_all_ta_features(
        data, open="open", high="high", low="low", close="close", volume="volume", fillna=True
    )
    
    # Dictionary to collect new columns
    new_columns = {}
    
    # Momentum Indicators
    rsi_14 = RSIIndicator(data['close'], window=14)
    new_columns['rsi_14'] = rsi_14.rsi()
    rsi_7 = RSIIndicator(data['close'], window=7)
    new_columns['rsi_7'] = rsi_7.rsi()
    
    stoch = StochasticOscillator(data['high'], data['low'], data['close'], window=14, smooth_window=3)
    new_columns['stochastic_k'] = stoch.stoch()
    new_columns['stochastic_d'] = stoch.stoch_signal()
    
    williams = WilliamsRIndicator(data['high'], data['low'], data['close'], lbp=14)
    new_columns['williams_r'] = williams.williams_r()
    
    roc_5 = ROCIndicator(data['close'], window=5)
    new_columns['roc_5'] = roc_5.roc()
    roc_10 = ROCIndicator(data['close'], window=10)
    new_columns['roc_10'] = roc_10.roc()
    
    # Trend Indicators
    ema_10 = EMAIndicator(data['close'], window=10)
    new_columns['ema_10'] = ema_10.ema_indicator()
    ema_20 = EMAIndicator(data['close'], window=20)
    new_columns['ema_20'] = ema_20.ema_indicator()
    
    sma_50 = SMAIndicator(data['close'], window=50)
    new_columns['sma_50'] = sma_50.sma_indicator()
    sma_200 = SMAIndicator(data['close'], window=200)
    new_columns['sma_200'] = sma_200.sma_indicator()
    
    macd = MACD(data['close'])
    new_columns['macd'] = macd.macd()
    new_columns['macd_signal'] = macd.macd_signal()
    
    adx = ADXIndicator(data['high'], data['low'], data['close'], window=14)
    new_columns['adx'] = adx.adx()
    new_columns['adx_pos'] = adx.adx_pos()
    new_columns['adx_neg'] = adx.adx_neg()
    
    # Volatility Indicators
    bb = BollingerBands(data['close'], window=20)
    new_columns['bollinger_mavg'] = bb.bollinger_mavg()
    new_columns['bollinger_hband'] = bb.bollinger_hband()
    new_columns['bollinger_lband'] = bb.bollinger_lband()
    new_columns['bollinger_width'] = (new_columns['bollinger_hband'] - new_columns['bollinger_lband']) / new_columns['bollinger_mavg']
    
    atr_14 = AverageTrueRange(data['high'], data['low'], data['close'], window=14)
    new_columns['atr_14'] = atr_14.average_true_range()
    atr_7 = AverageTrueRange(data['high'], data['low'], data['close'], window=7)
    new_columns['atr_7'] = atr_7.average_true_range()
    
    # Volume Indicators
    obv = OnBalanceVolumeIndicator(data['close'], data['volume'])
    new_columns['obv'] = obv.on_balance_volume()
    
    cmf = ChaikinMoneyFlowIndicator(data['high'], data['low'], data['close'], data['volume'], window=20)
    new_columns['cmf'] = cmf.chaikin_money_flow()
    
    vwap = VolumeWeightedAveragePrice(data['high'], data['low'], data['close'], data['volume'])
    new_columns['vwap'] = vwap.volume_weighted_average_price()
    
    # Custom Indicators
    psar = PSARIndicator(data['high'], data['low'], data['close'])
    new_columns['psar'] = psar.psar()
    
    ichimoku = IchimokuIndicator(data['high'], data['low'])
    new_columns['ichimoku_a'] = ichimoku.ichimoku_a()
    new_columns['ichimoku_b'] = ichimoku.ichimoku_b()
    new_columns['ichimoku_base'] = ichimoku.ichimoku_base_line()
    new_columns['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
    
    cci = CCIIndicator(data['high'], data['low'], data['close'], window=20)
    new_columns['cci'] = cci.cci()
    
    # Returns
    new_columns['return_1'] = data['close'].pct_change(1)
    new_columns['return_5'] = data['close'].pct_change(5)
    new_columns['return_20'] = data['close'].pct_change(20)
    
    # Concatenate all new columns at once
    new_df = pd.DataFrame(new_columns, index=data.index)
    data = pd.concat([data, new_df], axis=1)
    
    # Remove NaN values
    data = data.dropna()
    #logger.info(f"Indicators calculated, {len(data)} rows remaining")
    return data