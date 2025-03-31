import logging

def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(name)