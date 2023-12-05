import os
from typing import Iterator


def list_dir(directory_path: str) -> Iterator[str]:
    """
    Return directory contents with absolute path, recursively.
    :param directory_path: The directory to be listed
    :return: Iterator of file paths
    """
    for dirpath,_,filenames in os.walk(directory_path):
        for file_name in filenames:
            yield os.path.abspath(os.path.join(dirpath, file_name))
