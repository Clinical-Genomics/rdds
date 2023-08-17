from os.path import abspath, dirname


def get_workdir_path(module_name: str) -> str:
    """
    Provide path to module work directory
    :param module_name: Name of module (unique to calling module)
    :return: Path
    """
    workdir: str = abspath(dirname(__file__)+f'../../../../../tmp/{module_name}')
    return workdir
