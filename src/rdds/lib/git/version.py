from subprocess import run, CompletedProcess


def git_version() -> str:
    """
    Get the GIT version
    :return: Version as string
    NOTE: changes to this method should be reflected in the VERSION variable in Makefile
    """
    completed_process: CompletedProcess = run(args='git describe --tags --dirty --always',
                                              shell=True,
                                              capture_output=True,
                                              check=True)
    version: bytes = completed_process.stdout
    version: str = version.decode('utf-8')
    return version
