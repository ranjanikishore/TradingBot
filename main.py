import json
from data import CSVDataSource
from domain import CryptoTradingEnv, SACAgent
from application import Trainer
from visualization.visualization import plot_results
from utils.file_utils import get_unique_filename
from utils.logging import setup_logging

#logger = setup_logging(__name__)

def main():
    #logger.info("Starting main execution")
    try:
        # Data layer
        data_source = CSVDataSource()
        data = data_source.get_data()
        
        # Domain layer
        env = CryptoTradingEnv(data)
        #logger.info(f"Environment created with observation shape: {env.observation_space.shape}")
        agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0])
        #logger.info("Agent initialized, proceeding to training")
        
        # Application layer
        trainer = Trainer(env, agent)
        #logger.info("Trainer initialized, starting training")
        iteration_data = trainer.train()
        #logger.info("Training completed")
        
        # Save results
        filename = get_unique_filename("sac_trading_run")
        with open(filename, 'w') as f:
            json.dump(iteration_data, f, indent=4)
        
        # Save strategy history
        strategy_history_filename = get_unique_filename("strategy_history")
        with open(strategy_history_filename, 'w') as f:
            json.dump(env._CryptoTradingEnv__strategy_manager.history, f, indent=4)
        
        # Visualize results
        plot_results(iteration_data, env._CryptoTradingEnv__strategy_manager.history)
        
        #logger.info(f"Run completed. Data saved to {filename}, Strategy history saved to {strategy_history_filename}")
    except KeyboardInterrupt:
        #logger.info("Execution interrupted by user")
        if 'iteration_data' in locals():
            partial_filename = get_unique_filename("sac_trading_partial")
            with open(partial_filename, 'w') as f:
                json.dump(iteration_data, f, indent=4)
            #logger.info(f"Partial data saved to {partial_filename}")
        if 'env' in locals():
            partial_history_filename = get_unique_filename("strategy_history_partial")
            with open(partial_history_filename, 'w') as f:
                json.dump(env._CryptoTradingEnv__strategy_manager.history, f, indent=4)
            #logger.info(f"Partial strategy history saved to {partial_history_filename}")
        raise

if __name__ == "__main__":
    main()