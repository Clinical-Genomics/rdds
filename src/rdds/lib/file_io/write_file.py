from typing import List


def write_file(file_path: str, contents: List[str]):
    """
    Store contents to plain text file in file_path.
    :param file_path: Where to create the file
    :param content: Contents of file.
    """
    with open(file_path, 'w') as file_descriptor:
        for content in contents:
            print(content, file=file_descriptor)
