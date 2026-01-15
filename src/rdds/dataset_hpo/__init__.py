from os import makedirs

from rdds.lib.workdir import get_workdir_path

WORKDIR = get_workdir_path('dataset-hpo')

try:
    makedirs(WORKDIR)
except FileExistsError:
    pass