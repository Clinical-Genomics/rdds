import argparse
from time import time
from os import makedirs
from os.path import join
from typing import List

from . import WORKDIR
from .dataset import Dataset

parser = argparse.ArgumentParser(prog='Variant rank score model',
                                 description='Training API')
parser.add_argument('mode',
                    choices=['compile'])
parser.add_argument('--vcf',
                    help='VCF file to compile into dataset')
parser.add_argument('--vcf-mutacc-tp-cases',
                    default=None,
                    help='VCF containing MUTACC dump of positive cases')
parser.add_argument('--hd5',
                    help='Dataset HD5PY file to analyze')
parser.add_argument('--hd5ref',
                    help='Dataset HD5PY file to analyze reference')
parser.add_argument('--workdir',
                    default=WORKDIR)
parser.add_argument('--dataset-file-name',
                    default='dataset-%s.hdf5' % int(time()))
parser.add_argument('--features',
                    default=None,
                    help='VCF INFO.[NAME](s) to add to dataset, \',\' separated')
parser.add_argument('--image-name-prefix',
                    default=None,
                    help='Prefix to add to saved image names')
parser.add_argument('--k-fold-subset-size',
                    default=None,
                    help='Size of k fold subset for cross validation')
args = parser.parse_args()

try:
    makedirs(args.workdir)
except FileExistsError:
    pass

if args.mode == 'view':
    Dataset(args.hd5).view()
elif args.mode == 'compile':
    dataset: Dataset = Dataset(file_path=join(args.workdir, args.dataset_file_name))
    print(f'Compiling {args.vcf} -> {dataset.out_file_path}{args.features}')
    features: List[str] = args.features.split(',') if args.features is not None else None
    dataset.compile(vcf_file=args.vcf,
                    features=features)
    if args.vcf_mutacc_tp_cases is not None:
        print(f'Adding MUTACC true positive cases from {args.vcf_mutacc_tp_cases}')
        dataset.compile_structured_format_mutacc_tp_cases(mutacc_vcf_file_path=args.vcf_mutacc_tp_cases)
    dataset.view()
else:
    raise ValueError('Unknown op mode', args.mode)
