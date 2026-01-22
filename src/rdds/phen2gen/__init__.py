from os import makedirs

from rdds.lib.workdir import get_workdir_path

WORKDIR = get_workdir_path('pen2gen')

try:
    makedirs(WORKDIR)
except FileExistsError:
    pass

from rdds.lib.logging import get_logger
_LOGGER = get_logger(name='phen2gen', log_level='info')