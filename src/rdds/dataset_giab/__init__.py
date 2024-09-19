from os import makedirs

from rdds.lib.workdir import get_workdir_path

WORKDIR = get_workdir_path('dataset-giab')

try:
    makedirs(WORKDIR)
except FileExistsError:
    pass
