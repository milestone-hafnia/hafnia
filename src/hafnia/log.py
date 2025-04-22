import logging

from hafnia import __package_name__


class CustomFormatter(logging.Formatter):
    log_format = "%(asctime)s - %(name)s:%(filename)s @ %(lineno)d - %(levelname)s - %(message)s"

    def format(self, record):
        formatter = logging.Formatter(self.log_format)
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
