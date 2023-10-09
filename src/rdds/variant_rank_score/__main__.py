import argparse
from time import time
import os
from typing import List

from . import WORKDIR
from rdds.lib.hdf5 import Hdf5Viewer, Hd5DataGenerator
from rdds.lib.tf import get_tf_dataset_from_hd5_data_generator
from .model import VariantRankScoreModel
from .dataset import Dataset

parser = argparse.ArgumentParser(prog='Variant rank score model',
                                 description='Training API')
parser.add_argument('mode',
                    choices=['compile-vcf',
                             'train',
                             'view',
                             'predict'])
parser.add_argument('--vcf',
                    help='VCF file to compile into dataset')
parser.add_argument('--hd5',
                    help='Dataset HD5PY file to analyze')
parser.add_argument('--hd5ref',
                    help='Dataset HD5PY file to analyze reference')
parser.add_argument('--dataset-file-name',
                    default='dataset-%s.hdf5' % int(time()))
parser.add_argument('--saved-model')
args = parser.parse_args()

try:
    os.makedirs(WORKDIR)
except FileExistsError:
    pass

if args.mode == 'view':
    Hdf5Viewer(args.hd5)()
elif args.mode == 'compile-vcf':
    dataset: Dataset = Dataset(file_path=os.path.join(WORKDIR, args.dataset_file_name))
    print(f'Compiling {args.vcf} -> {dataset.out_file_path}')
    dataset.compile(vcf_file=args.vcf)
    dataset.view()
elif args.mode == 'train':
    variant_rank_score_model: VariantRankScoreModel = VariantRankScoreModel()
    variant_rank_score_model.train(args.hd5)
elif args.mode == 'predict':
    variant_rank_score_model: VariantRankScoreModel = VariantRankScoreModel()
    variant_rank_score_model.load_saved_model(model_path=args.saved_model)
    variant_rank_score_model.predict_on_hd5(args.hd5)
else:
    raise ValueError('Unknown op mode', args.mode)
