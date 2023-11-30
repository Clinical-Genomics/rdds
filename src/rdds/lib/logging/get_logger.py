from logging import Logger, basicConfig, getLogger, DEBUG, INFO, WARNING, FATAL

from rdds.lib.logging import constants

_LOG_LEVELS = {'debug': DEBUG,
               'info': INFO,
               'warning': WARNING,
               'fatal': FATAL}

def get_logger(name: str = None,
               log_level: str = None) -> Logger:
    """
    Convenience method for configuring logging module.
    :param name: Name of logger
    :param log_level: Str matching names in logging.[LOG_LEVEL]
    :return: An instance of Logger
    """
    basicConfig(format=constants.LOGGING_FORMAT)
    logger: Logger = getLogger(name=name)
    if log_level:
        logger.setLevel(_LOG_LEVELS[log_level])
    return logger
