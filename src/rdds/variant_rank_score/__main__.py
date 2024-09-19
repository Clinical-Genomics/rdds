import argparse
from time import time
import os

from . import WORKDIR
from rdds.lib.hdf5 import Hdf5Viewer
from .dataset import Dataset
from .dataset import VCFDataSet

try:
    os.makedirs(WORKDIR)
except FileExistsError:
    pass

parser = argparse.ArgumentParser(prog='Variant rank score model',
                                 description='Data management and model training CLI')
subparsers = parser.add_subparsers()
subparser = subparsers.add_parser('concat-vcfs', help='Concatenate VCFs to a single VCF')
subparser.add_argument('vcfs', nargs='+', help='List of paths to VCFs separated by ,')
subparser.set_defaults(func=lambda args: VCFDataSet().concat_datasets(args.vcfs))

subparser = subparsers.add_parser('compile-vcf', help='Compile a .vcf to .hd5 file')
subparser.add_argument('vcf', help='Path to VCF file')
subparser.add_argument('--dataset-file-name', default='dataset-%s.hdf5' % int(time()), help='output file name')
def compile_vcf(args):
    dataset: Dataset = Dataset(file_path=os.path.join(WORKDIR, args.dataset_file_name))
    print(f'Compiling {args.vcf} -> {dataset.out_file_path}')
    dataset.compile(vcf_file=args.vcf)
    dataset.view()
subparser.set_defaults(func=compile_vcf)

subparser = subparsers.add_parser('view', help='View .hd5 file')
subparser.add_argument('hd5', help='Path to .hd5 file')
subparser.set_defaults(func=lambda args: Hdf5Viewer(args.hd5)())

subparser = subparsers.add_parser('train', help='Run model training')
subparser.add_argument('hd5', help='Path to .hd5 file to be used as training, validation data')
def train(args):
    from .model import VariantRankScoreModel
    VariantRankScoreModel().train(args.hd5)
subparser.set_defaults(func=train)

subparser = subparsers.add_parser('predict', help='Load model and run inference on data')
subparser.add_argument('saved-model', help='Path to saved model directory')
subparser.add_argument('hd5', help='Path to .hd5 data file')
def predict(args):
    from .model import VariantRankScoreModel
    variant_rank_score_model: VariantRankScoreModel = VariantRankScoreModel()
    variant_rank_score_model.load_saved_model(model_path=args.saved_model)
    variant_rank_score_model.predict_on_hd5(args.hd5)
subparser.set_defaults(func=predict)

args = parser.parse_args()
args.func(args)  # Callback to trigger func with args
