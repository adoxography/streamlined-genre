"""
genre.util

Utility functions for streamlined-genre
"""
from math import ceil
from pathlib import Path

import wget


def get_project_root():
    """
    :return: the root directory of the project
    :rtype: Path
    """
    return Path(__file__).parent.parent


def ensure_download_exists(path, url):
    """
    Downloads a file to `path` if `path` does not exist

    :param path: The path where the file should exist
    :type path: Path
    :param url: The URL where the file should be downloaded from if it does not
                exist
    :type url: str
    """
    if not path.exists():
        wget.download(url, str(path))


def split_list(lst, percentage):
    """
    Splits `lst` by putting `percentage`% of the values in the first chunk,
    rounded up

    :param lst: The list to split
    :type lst: list
    :param percentage: The percentage of the elements that should be in the
                       first chunk
    :type percentage: float
    :return: Two lists, corresponding to the two chunks of the original list
    :rtype: (list, list)
    """
    split_point = ceil(len(lst) * percentage)
    return lst[:split_point], lst[split_point:]
