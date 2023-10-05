from logging import Logger, basicConfig, getLogger

from rdds.lib.logging import constants


def get_logger(name: str = None) -> Logger:
    """
    Convenience method for configuring logging module.
    :param name: Name of logger
    :return: An instance of Logger
    """
    basicConfig(format=constants.LOGGING_FORMAT)
    return getLogger(name=name)
