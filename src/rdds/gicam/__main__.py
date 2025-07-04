import argparse
from . import WORKDIR
from os import cpu_count
from glob import glob
from os.path import join
from progressbar import ProgressBar
from rdds.lib.git import git_version

parser = argparse.ArgumentParser(prog='GICAM',
                                 description='Module to merge MIVMIR and GENMOD inferences')
subparsers = parser.add_subparsers()

subparser = subparsers.add_parser('train', help='Train model')
subparser.add_argument('hd5_file',
                       help='Path to HD5 file containing MIVMIR, GENMOD and ground truth labels')
subparser.add_argument('--tune-hyperparams',
                       help='Tune model hyperparameters',
                       type=bool,
                       default=False)
subparser.add_argument('--hparam-max-epochs',
                       help='Maximum epochs per hparam run',
                       type=int,
                       default=10)
def train(args):
    from .model import Gicam
    from .hyperparameter_tuner import GicamBayesianTuner, HyperParameters

    workdir = join(WORKDIR, git_version())
    if args.tune_hyperparams:
        tuner = GicamBayesianTuner(hd5_file_path=args.hd5_file,
                                   log_dir=workdir,
                                   max_epochs=args.hparam_max_epochs)
        tuner.search_space_summary()
        tuner.search()
    else:
        gicam = Gicam(work_dir=workdir)
        gicam.build(path_to_hd5_dataset=args.hd5_file, hparams=HyperParameters())
        gicam.train()
        gicam.visualize_decision_boundary(storage_path='train-log-dir')
subparser.set_defaults(func=train)

subparser = subparsers.add_parser('build_export_train_data', help='Export training data')
subparser.add_argument('hd5_file',
                       help='Path to HD5 file containing MIVMIR, GENMOD and ground truth labels')
subparser.add_argument('output_file',
                       help='Export file path target')

def _export(args):
    from .dataset import DatasetLoader
    dataset_loader = DatasetLoader(path_to_hd5_dataset=args.hd5_file)
    dataset_loader.export_to_hd5(args.output_file)
subparser.set_defaults(func=_export)

subparser = subparsers.add_parser('explore', help='Visually explore hd5 file')
subparser.add_argument('hd5_file',
                       help='Path to HD5 file containing MIVMIR, GENMOD and ground truth labels')
def _explore(args):
    from .dataset.exploration import Explorer
    Explorer(path_to_hd5_dataset=args.hd5_file)()
subparser.set_defaults(func=_explore)

subparser = subparsers.add_parser('infer-vcf', help='Run model inference on VCF')
subparser.add_argument('vcf_file_path',
                       nargs='*',
                       help='Path to VCF file containing MIVMIR, GENMOD inferences. Globbing supported *.vcf')
subparser.add_argument('--cpu_cores',
                       help='Number of CPU cores to allocate for processing',
                       default=cpu_count() - 1)
subparser.add_argument('--replace_overwrite_vrs',
                       help='Write GICAM inference value to VrsModelPrediction field instead of separate GICAM' +
                            '(not to be used in production)',
                       default=False)
def _infer_vcf(args):
    from .vcf_inference import infer_vcf
    if '*' in args.vcf_file_path:
        # Globbing
        vcf_file_paths = glob(args.vcf_file_path)
    else:
        vcf_file_paths = args.vcf_file_path
    if len(vcf_file_paths) == 0:
        raise ValueError('No input VCF files. Expected at least one.')
    print(f'About to process files: {vcf_file_paths}')
    pbar = ProgressBar(max_value=len(vcf_file_paths))
    pbar.start()
    for vcf_file_path in vcf_file_paths:
        infer_vcf(vcf_file_path=vcf_file_path, cpu_cores=int(args.cpu_cores))
        pbar.increment(1)
    pbar.finish()
subparser.set_defaults(func=_infer_vcf)
args = parser.parse_args()
print(f"Called with args:\n{args}")
args.func(args)  # Callback to trigger func with args