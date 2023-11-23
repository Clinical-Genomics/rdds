import argparse
from rdds.lib.data_exploration import DataExplorer

parser = argparse.ArgumentParser()
parser.add_argument('hd5', help='hd5 file to analyse')

args = parser.parse_args()

data_explorer = DataExplorer(hd5_file_path=args.hd5)
data_explorer()
