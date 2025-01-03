import logging

from mdi_python_tools import __package_name__


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    log_format = "%(asctime)s - %(name)s:%(filename)s @ %(lineno)d - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + log_format + reset,
        logging.INFO: green + log_format + reset,
        logging.WARNING: yellow + log_format + reset,
        logging.ERROR: red + log_format + reset,
        logging.CRITICAL: bold_red + log_format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def create_logger() -> logging.Logger:
    root_logger = logging.getLogger(__package_name__)
    if root_logger.hasHandlers():
        return root_logger

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())

    root_logger.propagate = False
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

    root_logger.addHandler(ch)
    root_logger.setLevel(logging.INFO)
    return root_logger


logger = create_logger()
