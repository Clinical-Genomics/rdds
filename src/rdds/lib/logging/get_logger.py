import logging

from . import LOGGING_FORMAT


def get_logger(name: str = None) -> logging.Logger:
    """
    Convenience method for configuring logging module.
    :param name: Name of logger
    :return: An instance of Logger
    """
    logging.basicConfig(format=LOGGING_FORMAT)
    return logging.getLogger(name=name)
