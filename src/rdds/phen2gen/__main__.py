import argparse

from . import WORKDIR
from .dataset import Phen2GenDatasetCompiler

parser = argparse.ArgumentParser(prog='Phen2Gen Model',
                                 description='Data management and model training CLI')
subparsers = parser.add_subparsers()
subparser = subparsers.add_parser('compile-dataset', help='Build dataset for training')
subparser.set_defaults(func=lambda args: Phen2GenDatasetCompiler().compile())

subparsers = parser.add_subparsers()
subparser = subparsers.add_parser('build-train', help='Build dataset for training and train')
subparser.add_argument('--vcf', default='/rdds/tmp/justhusky_short.vcf', help='Path to VCF file')  # FIXME default
def build_train(args):
    from .model.model import GnnModel
    model = GnnModel()
    model.build_dataset(vcf_path=args.vcf_path)
subparser.set_defaults(func=lambda args: build_train(args))

args = parser.parse_args()
args.func(args)  # Callback to trigger func with args
