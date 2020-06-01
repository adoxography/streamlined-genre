"""
genre.util

Utility functions for streamlined-genre
"""
import itertools
from math import ceil
from pathlib import Path
from typing import Any, Callable, Iterable, List, Tuple

import wget  # type: ignore


def get_package_root() -> Path:
    """
    :return: the root directory of the project
    """
    return Path(__file__).parent


def get_config_dir() -> Path:
    """
    :return: The configuration directory
    """
    return get_package_root() / 'config'


def get_lib_dir() -> Path:
    """
    :return: The library directory
    """
    return get_package_root() / 'lib'


def ensure_download_exists(path: Path, url: str) -> None:
    """
    Downloads a file to `path` if `path` does not exist

    :param path: The path where the file should exist
    :param url: The URL where the file should be downloaded from if it does not
                exist
    """
    if not path.exists():
        wget.download(url, str(path))


def split_list(lst: List, percentage: float) -> Tuple[List, List]:
    """
    Splits `lst` by putting `percentage`% of the values in the first chunk,
    rounded up

    :param lst: The list to split
    :param percentage: The percentage of the elements that should be in the
                       first chunk
    :return: Two lists, corresponding to the two chunks of the original list
    """
    split_point = ceil(len(lst) * percentage)
    return lst[:split_point], lst[split_point:]


def first(iterable: Iterable, key: Callable[[Any], bool]) -> Any:
    """
    Finds the first element in `iterable` that satisfies `key`

    :param iterable: The iterable to search
    :param key: The function to check iterable with
    :return: The first value that satisfies `key`

    :raises ValueError: No element in `iterable` satisfies `key`
    """
    try:
        return next(x for x in iterable if key(x))
    except StopIteration:
        raise ValueError('No element matched')


def flatten(iterable: Iterable) -> List:
    """
    flattens one layer of a nested iterable

    :param iterable: The iterable to flatten
    :return: A flattened version of the iterable
    """
    return list(itertools.chain(*iterable))
