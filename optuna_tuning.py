import optuna
import pandas as pd
from config import Config
from data.data_source import CSVDataSource
from domain.environment import TradingEnvironment
from domain.agent import TradingAgent
from application.trainer import Trainer
from utils.logging import setup_logging

#logger = setup_logging(__name__)

def objective(trial):
    Config.LEARNING_RATE = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    Config.BATCH_SIZE = trial.suggest_int('batch_size', 32, 256)
    Config.ENTROPY_TARGET_MULTIPLIER = trial.suggest_float('entropy_multiplier', 0.1, 2.0)
    
    data_source = CSVDataSource("bitcoin_data_cleaned.csv")
    data = data_source.get_data()
    env = TradingEnvironment(data)
    state_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[0]
    agents = [TradingAgent(state_dim, action_dim)]
    trainer = Trainer(env, agents)
    iteration_data = trainer.train()
    final_value = iteration_data[-1]['portfolio_value']
    return final_value

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    #logger.info(f"Best hyperparameters: {study.best_params}")