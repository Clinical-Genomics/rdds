from subprocess import run, CompletedProcess, CalledProcessError

WORKTREE_VERSION_TOKEN: str = 'unknown-worktree-version'


def _git_version() -> str:
    """
    Get the GIT version
    :return: Version as string
    NOTE: changes to this method should be reflected in the VERSION variable in Makefile
    """
    run(f'git config --global --add safe.directory /rdds', shell=True, check=True, capture_output=True)  # Fix for git unsafe directory error
    completed_process: CompletedProcess = run(args='git describe --tags --dirty --always',
                                              shell=True,
                                              capture_output=True,
                                              check=True)
    version: bytes = completed_process.stdout
    version: str = version.decode('utf-8').replace('\n', '')
    return version

def git_version() -> str:
    """
    Get the GIT version as string.
    In case repository is a worktree instance and git fails, return WORKTREE_VERSION_TOKEN as version since
    the parent .git directory is inaccessible (not mounted in container).
    :return: version string OR WORKTREE_VERSION_TOKEN
    """
    try:
        version = _git_version()
        return version
    except Exception as e:
        if e.stderr is not None:
            msg = e.stderr.decode('utf-8')
            if 'fatal: not a git repository' in msg and \
                '.git/worktrees' in msg:
                return WORKTREE_VERSION_TOKEN
        raise e