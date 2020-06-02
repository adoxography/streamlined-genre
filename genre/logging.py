"""
genre.logging

Handles interfacing between the library and its logger
"""
import logging
from pathlib import Path


logger = logging.getLogger('streamlined-genre')
logger.setLevel(logging.INFO)


def create_handler(path: Path) -> logging.FileHandler:
    """
    Creates a log handler

    :param path: The name of the file that should hold the log
    :return: The FileHandler
    """
    handler = logging.FileHandler(path)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # noqa
    handler.setFormatter(formatter)

    return handler
