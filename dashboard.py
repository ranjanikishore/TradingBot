import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import torch  # Added for torch.load
from datetime import datetime
import time
import threading
import queue

from sac_trading import CryptoTradingEnv, SACAgent, CSVDataSource, MockDataSource

data_queue = queue.Queue()
trading_active = threading.Event()
trading_active.clear()
current_env = None
current_agent = None
performance_history = []

def run_trading_bot(data_source_type, initial_balance, stop_loss, max_position):
    """Run the trading bot in a separate thread"""
    global current_env, current_agent, performance_history
    if data_source_type == "csv":
        data_source = CSVDataSource()
    else:
        data_source = MockDataSource()
    print(f"Starting bot with data source: {data_source_type}, balance: ${initial_balance}")
    try:
        current_env = CryptoTradingEnv(data_source, initial_balance=initial_balance)
    except ValueError as e:
        print(f"Failed to initialize environment: {str(e)}. Switching to mock data")
        current_env = CryptoTradingEnv(MockDataSource(), initial_balance=initial_balance)
    current_env.stop_loss = stop_loss
    current_env.max_position = max_position
    current_agent = SACAgent(current_env.observation_space.shape[0], current_env.action_space.shape[0])
    try:
        current_agent.actor.load_state_dict(torch.load("sac_btc_trader.pth"))
        print("Loaded pre-trained model")
    except FileNotFoundError:
        print("Warning: No pre-trained model found at 'sac_btc_trader.pth'. Starting fresh")
    state = current_env.reset()
    performance_history = []
    while True:
        if not trading_active.is_set():
            print("Trading paused")
            time.sleep(1)
            continue
        try:
            action = current_agent.select_action(state, deterministic=True)
            next_state, reward, done, info = current_env.step(action)
            performance_data = {
                'timestamp': current_env.data.iloc[current_env.current_step]['timestamp'],
                'portfolio_value': info['value'],
                'action': action[0],
                'reward': reward,
                'volatility': info['volatility'],
                'btc_price': current_env.data.iloc[current_env.current_step]['close'],
                'rsi': current_env.data.iloc[current_env.current_step]['rsi'],
                'macd': current_env.data.iloc[current_env.current_step]['macd'],
                'atr': current_env.data.iloc[current_env.current_step]['atr']
            }
            performance_history.append(performance_data)
            data_queue.put(performance_data)
            current_agent.replay_buffer.push(state, action, reward, next_state, done)
            current_agent.train()
            state = next_state
            if done:
                state = current_env.reset()
                print("Reached end of data, resetting environment")
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in trading loop: {str(e)}")
            time.sleep(1)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Enhanced Crypto Trading Bot Dashboard"),
    html.Div([
        html.H3("Control Panel"),
        html.Label("Data Source:"),
        dcc.Dropdown(id='data-source-dropdown', options=[
            {'label': 'CSV (bitcoin_data.csv)', 'value': 'csv'},
            {'label': 'Mock Data', 'value': 'mock'}
        ], value='csv'),
        html.Label("Initial Balance ($):"),
        dcc.Input(id='initial-balance', type='number', value=10000, min=1000, step=1000),
        html.Label("Stop Loss (%):"),
        dcc.Slider(id='stop-loss', min=1, max=20, step=1, value=5, marks={i: f'{i}%' for i in range(1, 21, 5)}),
        html.Label("Max Position (%):"),
        dcc.Slider(id='max-position', min=10, max=100, step=10, value=50, marks={i: f'{i}%' for i in range(10, 101, 20)}),
        html.Button('Start Trading', id='start-button', n_clicks=0),
        html.Button('Stop Trading', id='stop-button', n_clicks=0),
        html.Div(id='trading-status', children="Trading Stopped")
    ], style={'padding': '20px', 'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    html.Div([
        html.H3("Live Metrics"),
        html.Div([html.Div("Portfolio Value: ", style={'display': 'inline-block'}),
                  html.Div(id='live-portfolio-value', style={'display': 'inline-block', 'marginLeft': '10px'})]),
        html.Div([html.Div("Current Action: ", style={'display': 'inline-block'}),
                  html.Div(id='live-action', style={'display': 'inline-block', 'marginLeft': '10px'})]),
        html.Div([html.Div("BTC Price: ", style={'display': 'inline-block'}),
                  html.Div(id='live-btc-price', style={'display': 'inline-block', 'marginLeft': '10px'})]),
        html.Div([html.Div("Volatility: ", style={'display': 'inline-block'}),
                  html.Div(id='live-volatility', style={'display': 'inline-block', 'marginLeft': '10px'})]),
        dcc.Graph(id='live-performance-graph'),
        dcc.Graph(id='technical-indicators-graph'),
        html.H3("Historical Performance"),
        dcc.Graph(id='historical-performance-graph'),
        html.H3("Performance Metrics"),
        dcc.Graph(id='performance-metrics-graph'),
        dcc.Interval(id='interval-component', interval=100, n_intervals=0)
    ], style={'padding': '20px', 'width': '65%', 'display': 'inline-block'})
])

historical_data = None
try:
    csv_data = CSVDataSource().get_data()
    if csv_data.empty:
        historical_data = MockDataSource().get_data()
        print("Using mock data for historical display due to empty CSV")
    else:
        historical_data = csv_data
        print("Using CSV data for historical display")
except Exception as e:
    historical_data = MockDataSource().get_data()
    print(f"CSV loading failed ({str(e)}), using mock data for historical display")

@app.callback(
    Output('trading-status', 'children'),
    [Input('start-button', 'n_clicks'), Input('stop-button', 'n_clicks')],
    [State('data-source-dropdown', 'value'), State('initial-balance', 'value'),
     State('stop-loss', 'value'), State('max-position', 'value')]
)
def control_trading(start_clicks, stop_clicks, data_source, initial_balance, stop_loss, max_position):
    """Handle start/stop trading controls"""
    global trading_thread
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Trading Stopped"
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'start-button':
        if not trading_active.is_set():
            trading_active.set()
            trading_thread = threading.Thread(
                target=run_trading_bot,
                args=(data_source, initial_balance, stop_loss/100, max_position/100),
                daemon=True
            )
            trading_thread.start()
            print("Trading started")
            return "Trading Active"
    elif button_id == 'stop-button':
        trading_active.clear()
        print("Trading stopped")
        return "Trading Stopped"
    return "Trading Stopped"

@app.callback(
    [Output('live-portfolio-value', 'children'), Output('live-action', 'children'),
     Output('live-btc-price', 'children'), Output('live-volatility', 'children'),
     Output('live-performance-graph', 'figure'), Output('technical-indicators-graph', 'figure'),
     Output('performance-metrics-graph', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_live_metrics(n):
    """Update live metrics and graphs with latest data"""
    global performance_history
    performance_data = {'portfolio_value': 10000, 'action': 0, 'btc_price': 0, 'volatility': 0,
                       'rsi': 0, 'macd': 0, 'atr': 0}
    max_items = 100  # Limit queue processing
    for _ in range(min(max_items, data_queue.qsize())):
        if not data_queue.empty():
            data = data_queue.get()
            performance_history.append(data)
            performance_data = data
    portfolio_value = f"${performance_data['portfolio_value']:,.2f}"
    action = f"{performance_data['action']:.2f}"
    btc_price = f"${performance_data['btc_price']:,.2f}"
    volatility = f"{performance_data['volatility']:.4f}"
    timestamps = [data['timestamp'] for data in performance_history[-100:]]
    values = [data['portfolio_value'] for data in performance_history[-100:]]
    perf_figure = {
        'data': [{'x': timestamps, 'y': values, 'type': 'scatter', 'name': 'Portfolio Value'}],
        'layout': {'title': 'Live Portfolio Performance', 'xaxis': {'title': 'Time'}, 'yaxis': {'title': 'Value ($)'}}
    }
    rsi = [data['rsi'] for data in performance_history[-100:]]
    macd = [data['macd'] for data in performance_history[-100:]]
    atr = [data['atr'] for data in performance_history[-100:]]
    tech_figure = {
        'data': [
            {'x': timestamps, 'y': rsi, 'type': 'scatter', 'name': 'RSI', 'yaxis': 'y1'},
            {'x': timestamps, 'y': macd, 'type': 'scatter', 'name': 'MACD', 'yaxis': 'y2'},
            {'x': timestamps, 'y': atr, 'type': 'scatter', 'name': 'ATR', 'yaxis': 'y3'}
        ],
        'layout': {
            'title': 'Technical Indicators', 'xaxis': {'title': 'Time'},
            'yaxis': {'title': 'RSI', 'domain': [0, 0.3]},
            'yaxis2': {'title': 'MACD', 'domain': [0.35, 0.65], 'overlaying': 'y'},
            'yaxis3': {'title': 'ATR', 'domain': [0.7, 1.0], 'overlaying': 'y'}
        }
    }
    if len(performance_history) > 1:
        returns = np.diff([data['portfolio_value'] for data in performance_history])
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        max_drawdown = min(0, min(np.diff([data['portfolio_value'] for data in performance_history])) / performance_history[0]['portfolio_value'])
        metrics_figure = {
            'data': [go.Bar(x=['Sharpe Ratio', 'Max Drawdown'], y=[sharpe, max_drawdown],
                           text=[f'{sharpe:.2f}', f'{max_drawdown:.2%}'], textposition='auto')],
            'layout': {'title': 'Performance Metrics', 'yaxis': {'title': 'Value'}}
        }
    else:
        metrics_figure = {'data': [], 'layout': {'title': 'Performance Metrics (Insufficient Data)'}}
    print(f"Updated dashboard: Value={portfolio_value}, Action={action}")
    return portfolio_value, action, btc_price, volatility, perf_figure, tech_figure, metrics_figure

@app.callback(
    Output('historical-performance-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_historical_graph(n):
    """Update historical performance graph"""
    if historical_data is None or historical_data.empty:
        print("No historical data available")
        return {'data': [], 'layout': {'title': 'No Historical Data Available'}}
    figure = {
        'data': [{'x': historical_data['timestamp'], 'y': historical_data['close'], 'type': 'scatter', 'name': 'BTC Price'}],
        'layout': {'title': 'Historical BTC Price (5 Years Hourly)', 'xaxis': {'title': 'Time'}, 'yaxis': {'title': 'Price ($)'}}
    }
    return figure

if __name__ == "__main__":
    print("Starting Dash server")
    app.run(debug=True, use_reloader=False)
