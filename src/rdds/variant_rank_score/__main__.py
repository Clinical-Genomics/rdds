import argparse
from time import time
import os
import gc

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

subparser = subparsers.add_parser('compile-vcf',
                                   help='Compile a .vcf to .hd5 file. All of VCF dataset must fit in available RAM.')
subparser.add_argument('vcf', help='Path to VCF file')
subparser.add_argument('--dataset-file-name', default='dataset-%s.hdf5' % int(time()), help='output file name')
subparser.add_argument('--n-workers', default=os.cpu_count(), help='Maximum amount of concurrent workers (processes) allowed')
def compile_vcf(args):
    dataset: Dataset = Dataset(file_path=os.path.join(WORKDIR, args.dataset_file_name),
                               max_n_workers=int(args.n_workers))
    print(f'Compiling {args.vcf} -> {dataset.out_file_path}')
    dataset.compile(vcf_file=args.vcf)
    dataset.view()
subparser.set_defaults(func=compile_vcf)

subparser = subparsers.add_parser('view', help='View .hd5 file')
subparser.add_argument('hd5', help='Path to .hd5 file')
subparser.set_defaults(func=lambda args: Hdf5Viewer(args.hd5)())

subparser = subparsers.add_parser('train', help='Run model training')
subparser.add_argument('hd5', help='Path to .hd5 file to be used as training, validation data')
subparser.add_argument('--compile-vocabulary-normalisation-factors',
                       help='Generate new vocabulary file and normalisation factors from supplied data',
                       type=bool,
                       default=False)
subparser.add_argument('--tune-hyperparams',
                       help='Tune model hyperparameters',
                       type=bool,
                       default=False)
def train(args):
    from .model import VariantRankScoreModel
    from .hyperparameter_tuner import VRSBayesianTuner, HyperParameters

    if args.tune_hyperparams:
        tuner = VRSBayesianTuner(hd5_file_path=args.hd5,
                                 log_dir=WORKDIR)
        tuner.search_space_summary()
        tuner.search()
    else:
        model = VariantRankScoreModel()
        model.build(hd5_file_path=args.hd5,
                    hparams=model.get_uninitialized_hyperparameters(),
                    compile_vocabulary_normalisation_factors=args.compile_vocabulary_normalisation_factors)
        model.train()
subparser.set_defaults(func=train)

subparser = subparsers.add_parser('inference_exploration', help='Visualize model performance on .hd5 dataset')
subparser.add_argument('saved_model', help='Path to saved model directory')
subparser.add_argument('hd5', help='Path to .hd5 data file containing data for computing inferences')
def inference_exploration(args):
    from .model import VariantRankScoreModel
    from .inference_exploration import InferenceExplorer
    variant_rank_score_model: VariantRankScoreModel = VariantRankScoreModel()
    variant_rank_score_model.load_saved_model(model_path=args.saved_model)
    inferences_file_path = variant_rank_score_model.predict_on_hd5(args.hd5)
    del variant_rank_score_model
    gc.collect()
    inference_explorer = InferenceExplorer(hd5_file_path=inferences_file_path)
    inference_explorer()
subparser.set_defaults(func=inference_exploration)

subparser = subparsers.add_parser('predict-on-vcf', help='Run pretrained model on VCF to generate inferences')
subparser.add_argument('pretrained_model_path', help='Path to VRS pretrained model to load')
subparser.add_argument('vcf_file_path', help='Path to VCF file to generate inferences for')
subparser.add_argument('--cpu_cores',
                       help='Number of CPU cores to allocate for processing',
                       default=os.cpu_count() - 1)
def predict_on_vcf(args):
    from .vcf_inference import predict_on_vcf
    predict_on_vcf(vrs_model_file_path=args.pretrained_model_path,
                   vcf_file_path=args.vcf_file_path,
                   cpu_cores=int(args.cpu_cores))
subparser.set_defaults(func=predict_on_vcf)

args = parser.parse_args()
args.func(args)  # Callback to trigger func with args
