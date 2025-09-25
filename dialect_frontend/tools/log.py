from loguru import logger
import sys


class Log:
    def __init__(self):
        self.format = "<g>{time:MM-DD HH:mm:ss}</g> <lvl>{level:<9}</lvl>| {file}:{line} | {message}"

    def set_format(self, format_string):
        self.format = format_string
        logger.configure(format=format_string)

    @staticmethod
    def set_level(level: str):
        logger.remove()
        format = "<g>{time:MM-DD HH:mm:ss}</g> <lvl>{level:<9}</lvl>| {file}:{line} | {message}"
        logger.add(sys.stderr, format=format, level=level)

    @staticmethod
    def debug(message):
        return logger.debug(message)

    @staticmethod
    def info(message):
        return logger.info(message)

    @staticmethod
    def warning(message):
        return logger.warning(message)

    @staticmethod
    def error(message):
        return logger.error(message)

    @staticmethod
    def critical(message):
        return logger.critical(message)
