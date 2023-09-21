import argparse
from time import time
from os import makedirs
from os.path import join
from typing import List

from . import WORKDIR
from .dataset import Dataset
from .rankscore_normalization_tests import run_rankscore_normalization_tests
from .rankscore_stats import rankscore_stats

parser = argparse.ArgumentParser(prog='Rankscore exploration tool',
                                 description='Analyze VCF with respect to rank score')
parser.add_argument('mode',
                    choices=['compile',
                             'testnormalizedrankscore',
                             'view',
                             'rankscorestats'],
                    help='Processing mode')
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
                    default='RankScore',
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
    features: List[str] = args.features.split(',')
    print(f'Compiling {args.vcf} -> {dataset.file_path}{features}')
    dataset.compile(vcf_files=[args.vcf],
                    features=features)
    if args.vcf_mutacc_tp_cases is not None:
        print(f'Adding MUTACC true positive cases from {args.vcf_mutacc_tp_cases}')
        dataset.compile_structured_format_mutacc_tp_cases(mutacc_vcf_file_path=args.vcf_mutacc_tp_cases)
elif args.mode == 'testnormalizedrankscore':
    Dataset(args.hd5).view()
    if args.hd5ref:
        Dataset(args.hd5ref).view()
    run_rankscore_normalization_tests(file_path=args.hd5, file_path_ref=args.hd5ref)
elif args.mode == 'rankscorestats':
    Dataset(args.hd5).view()
    k_fold_subset_size = int(args.k_fold_subset_size) if args.k_fold_subset_size is not None else None
    rankscore_stats(file_path=args.hd5, image_name_prefix=args.image_name_prefix, k_fold_subset_size=k_fold_subset_size)
else:
    raise ValueError('Unknown op mode', args.mode)
