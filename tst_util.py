import logging


def get_logger(name, log_file):
    logger = logging.getLogger(name)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s, %(funcName)s %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger
