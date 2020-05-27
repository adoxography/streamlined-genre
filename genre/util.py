"""
genre.util

Utility functions for streamlined-genre
"""
import wget


def ensure_download_exists(path, url):
    """ Downloads a file to `path` if `path` does not exist """
    if not path.exists():
        wget.download(url, path)
