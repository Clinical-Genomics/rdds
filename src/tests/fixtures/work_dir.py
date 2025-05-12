import pytest
import tempfile
import shutil

@pytest.fixture
def work_dir() -> str:
    """
    Return a temporary working directory.
    :return: A path
    """
    work_dir = tempfile.mkdtemp(dir='/tmp')
    yield work_dir
    shutil.rmtree(work_dir)
