import os.path
from subprocess import run, CompletedProcess, CalledProcessError

from rdds.lib.logging import get_logger
WORKTREE_VERSION_TOKEN: str = 'unknown-worktree-version'

_LOGGER = get_logger('git', 'info')

GITCONFIG_PATH = '/root/.gitconfig'  # Global git config file


def _fix_git_unsafe_error():
    # Fix for git unsafe directory error
    if not os.path.exists(GITCONFIG_PATH):
        _LOGGER.info(f"Created global git config file {GITCONFIG_PATH}")
        run(f"touch {GITCONFIG_PATH}", shell=True, check=True)
    completed_process = run('git config --global -l', shell=True, capture_output=True, check=True)
    git_global_config = completed_process.stdout.decode('utf-8')
    if not 'safe.directory=/rdds' in git_global_config:
        _LOGGER.info('Adding /rdds as safe directory to global git config')
        run(f'git config --global --add safe.directory /rdds', shell=True, check=True, capture_output=True)


def _git_version() -> str:
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
    version: str = version.decode('utf-8').replace('\n', '')
    return version


def git_version() -> str:
    """
    Get the GIT version as string.
    In case repository is a worktree instance and git fails, return WORKTREE_VERSION_TOKEN as version since
    the parent .git directory is inaccessible (not mounted in container).
    :return: version string OR WORKTREE_VERSION_TOKEN
    """
    _fix_git_unsafe_error()
    try:
        version = _git_version()
        return version
    except CalledProcessError as e:
        if e.stderr is not None:
            msg = e.stderr.decode('utf-8')
            if 'fatal: not a git repository' in msg and \
                '.git/worktrees' in msg:
                return WORKTREE_VERSION_TOKEN
        raise e