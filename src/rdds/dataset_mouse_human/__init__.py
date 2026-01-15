from os import makedirs

from rdds.lib.workdir import get_workdir_path

WORKDIR = get_workdir_path('dataset-mouse-human')

try:
    makedirs(WORKDIR)
except FileExistsError:
    pass