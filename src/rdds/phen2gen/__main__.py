import argparse

from . import WORKDIR
from .dataset import Phen2GenDatasetCompiler

parser = argparse.ArgumentParser(prog='Phen2Gen Model',
                                 description='Data management and model training CLI')
subparsers = parser.add_subparsers()
subparser = subparsers.add_parser('compile-dataset', help='Build dataset for training')
subparser.set_defaults(func=lambda args: Phen2GenDatasetCompiler().compile())

args = parser.parse_args()
args.func(args)  # Callback to trigger func with args
