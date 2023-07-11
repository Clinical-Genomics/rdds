import argparse
from rdds.lib.hdf5.viewer import Hdf5Viewer

parser = argparse.ArgumentParser(prog='HDF5 dataset tool',
                                 description='Views HDF5 datasets structure')
parser.add_argument('input',
                    help='Path to VCF file to convert')

args = parser.parse_args()
Hdf5Viewer(file_path=args.input)()
