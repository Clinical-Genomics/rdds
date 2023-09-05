import argparse
from rdds.lib.hdf5.viewer import Hdf5Viewer

parser = argparse.ArgumentParser(prog='HDF5 dataset tool',
                                 description='Views HDF5 datasets structure')
parser.add_argument('hd5file',
                    help='Path to HD5File to view')

args = parser.parse_args()
Hdf5Viewer(file_path=args.hd5file)()
