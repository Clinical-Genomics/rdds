from subprocess import check_call

from rdds.lib.git import git_version


def test_version():
    """
    Test for git version, making sure its a valid tag in GIT.
    """
    # GIVEN a version
    # WHEN calling get version
    version = git_version()
    if not version:
        # if test is running in a git worktree, git version is inaccesible
        return
    # THEN make sure it's a valid tag in the repo
    assert isinstance(version, str)
    assert len(version) != 0
    check_call(f'git show {version.replace("-dirty", "")}', shell=True)
