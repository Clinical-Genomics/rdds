from hashlib import new as new_digest


def checksum(file_path: str,
             algorithm: str) -> str:
    """
    Compute checksum of file in binary mode.
    :param file_path: Path to file
    :param algorithm: Checksum algorithm, md5, sha256 etc
    :return: Hex digest of file
    """
    with open(file_path, 'rb') as file:
        digest = new_digest(algorithm)
        digest.update(file.read())
    return digest.hexdigest()
