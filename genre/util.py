"""
genre.util

Utility functions for streamlined-genre
"""
from math import ceil
from pathlib import Path

import wget


def get_project_root():
    """ Retrieves the root directory of the project """
    return Path(__file__).parent.parent


def ensure_download_exists(path, url):
    """ Downloads a file to `path` if `path` does not exist """
    if not path.exists():
        wget.download(url, str(path))


def split_list(lst, percentage):
    """
    Splits `lst` by putting `percentage`% of the values in the first half,
    rounded up
    """
    split_point = ceil(len(lst) * percentage)
    return lst[:split_point], lst[split_point:]
