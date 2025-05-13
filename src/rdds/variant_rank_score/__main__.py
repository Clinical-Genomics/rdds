import argparse
from time import time
import os
import gc
from typing import List
from glob import glob

from . import WORKDIR
from rdds.lib.hdf5 import Hdf5Viewer
from .dataset import Dataset
from .dataset import VCFDataSet
from .model.default_model import DEFAULT_MODEL_SPEC

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
subparser.add_argument('--extensive_training_metrics',
                       help='Add additional performance metrics to stratify variant performance (debug)',
                       type=bool,
                       default=False)
subparser.add_argument('--fast_debug_init',
                       help='Throttle data set size and shuffling to improve init, training speed (debug)',
                       type=bool,
                       default=False)
subparser.add_argument('--epochs',
                       help='Number of training epochs',
                       type=int,
                       default=None)
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
                    compile_vocabulary_normalisation_factors=args.compile_vocabulary_normalisation_factors,
                    extensive_training_metrics=args.extensive_training_metrics,
                    fast_debug_init=args.fast_debug_init)
        model.train(train_epochs=args.epochs)
        model.train_model_explainer()
subparser.set_defaults(func=train)

subparser = subparsers.add_parser('inference_exploration', help='Visualize model performance on .hd5 dataset')
subparser.add_argument('saved_model_path', help='Path to keras saved model (*.keras)')
subparser.add_argument('saved_model_explainer_path', help='Path to saved model explainer (model-explainer.bin)')
subparser.add_argument('hd5', help='Path to .hd5 data file containing data for computing inferences')
def inference_exploration(args):
    from .model import VariantRankScoreModel
    from .inference_exploration import InferenceExplorer
    variant_rank_score_model: VariantRankScoreModel = VariantRankScoreModel()
    variant_rank_score_model.load_saved_model(keras_model_path=args.saved_model_path,
                                              model_explainer_path=args.saved_model_explainer_path)
    inferences_file_path = variant_rank_score_model.predict_on_hd5(args.hd5)
    del variant_rank_score_model
    gc.collect()
    inference_explorer = InferenceExplorer(hd5_file_path=inferences_file_path)
    inference_explorer()
subparser.set_defaults(func=inference_exploration)

subparser = subparsers.add_parser('predict-on-vcf', help='Run pretrained model on VCF to generate inferences')
subparser.add_argument('vcf_file_path', nargs='*',
                       help='Path to ranked VCF to analyze. [CASENAME_suffix[es].vcf]. Globbing supported *.vcf')
subparser.add_argument('--pretrained_model_path', help='Path to VRS pretrained model to load',
                       default=DEFAULT_MODEL_SPEC.keras_model)
subparser.add_argument('--pretrained_model_explainer_path', help='Path to VRS pretrained ModelExplainer to load',
                       default=DEFAULT_MODEL_SPEC.explainer_model)
subparser.add_argument('--cpu_cores',
                       help='Number of CPU cores to allocate for processing',
                       default=os.cpu_count() - 1)
def predict_on_vcf(args):
    from .vcf_inference import predict_on_vcf
    if '*' in args.vcf_file_path:
        # Globbing
        vcf_file_paths = glob(args.vcf_file_path)
    else:
        vcf_file_paths = args.vcf_file_path
    if len(vcf_file_paths) == 0:
        raise ValueError('No input VCF files. Expected at least one.')
    print(f'About to process files: {vcf_file_paths}')
    for vcf_file_path in vcf_file_paths:
        predict_on_vcf(vrs_model_file_path=args.pretrained_model_path,
                       model_explainer_path=args.pretrained_model_explainer_path,
                       vcf_file_path=vcf_file_path,
                       cpu_cores=int(args.cpu_cores))
subparser.set_defaults(func=predict_on_vcf)

subparser = subparsers.add_parser('post-train-explainer', help='Train explanations model from pretrained keras model')
subparser.add_argument('pretrained_model_path', help='Path to keras model')
subparser.add_argument('hd5_file_path', help='HD5 Dataset file')
def post_train_explainer(args):
    from .model import VariantRankScoreModel
    model = VariantRankScoreModel()
    model.post_train_explainer(model_path=args.pretrained_model_path,
                               hd5_file_path=args.hd5_file_path)
subparser.set_defaults(func=post_train_explainer)

subparser = subparsers.add_parser('view-ranked-vcf', help='Visualize ranked variants in VCF')
subparser.add_argument('vcf_file_path', nargs='*',
                       help='Path to ranked VCF to analyze. [CASENAME_suffix[es].vcf]. Globbing supported *-predictions.vcf')
subparser.add_argument('--pathogenic_vcf_suffix',
                       help='Suffix identifying pathogenic variant VCF file',
                       default='_pathogenic.vcf')
def view_ranked_vcf(args):
    """
    Analyse real case VCF files and view model performance.
    Input files are ordered as such:
    - [CATALOG]
        - [case_name]_gatkcomb_rhocall_norm_af_mt_frqf_cadd_vep_parsed_ranked-[model_inference_suffix; -predictions].vcf
            (all case variants, annotated)
        - [case_name]_pathogenic.vcf
            (MUTACC confirmed positive pathogenic variant(s) for this particular case).
            Excerpt from above complete case data file.
        - ... Additional files
    Multiple files can be selected at the same time, by globbing on commandline: ./test_cases/*-predictions.vcf
    """
    from .inference_exploration import view_vcf_rank_results, aggregate_vcf_rank_results
    case_names: List[str] = []
    if len(args.vcf_file_path) == 1 and '*' in args.vcf_file_path[0]:
        # Globbing
        vcf_file_paths = glob(args.vcf_file_path[0])
    else:
        vcf_file_paths = args.vcf_file_path
    print(f'About to process files: {vcf_file_paths}')
    for vcf_file_path in vcf_file_paths:
        print(f'Processing {vcf_file_path}')
        # Case name is the initial prefix in the VCF file name, separated by underscore
        case_name = os.path.basename(vcf_file_path).split('_')[0]
        case_names.append(case_name)
        vcf_pathogenic_path = os.path.join(os.path.dirname(vcf_file_path), f'{case_name}{args.pathogenic_vcf_suffix}')
        work_dir = os.path.dirname(vcf_file_path)
        output_dir = os.path.join(work_dir, os.path.basename(vcf_file_path).split('_')[0])  # Case name as sub dir
        view_vcf_rank_results(vcf_file_path=vcf_file_path,
                              vcf_pathogenic_path=vcf_pathogenic_path,
                              workdir=output_dir)
        gc.collect()
    aggregate_vcf_rank_results(view_rank_result_output_dir=work_dir,
                               case_names=case_names)
subparser.set_defaults(func=view_ranked_vcf)


subparser = subparsers.add_parser('pipeline-performance-test', help='Profile data pipeline and view results in Tensorboard')
subparser.add_argument('hd5', help='Path to .hd5 file to be used as training, validation data')
subparser.add_argument('--batches', help='Number of batches to profile', default=int(10))
subparser.add_argument('--include_dataset_init', help='Include dataset bootstrapping (biasing result)', default=False)
subparser.add_argument('--start_on_first_epoch_end', help='Start profiling pipeline once all data is cached', default=True)
def profile_data_pipeline(args):
    import os
    from math import ceil
    from progressbar import ProgressBar
    from .model import VariantRankScoreModel
    from rdds.lib.tf.profiler import TfProfiler, Trace
    batches = int(args.batches)
    model = VariantRankScoreModel()
    hparams = model.get_uninitialized_hyperparameters()
    hparams.Int('batch_size', min_value=32, max_value=128, default=64)
    model._init_datasets(hd5_file_path=args.hd5,
                         hparams=hparams,
                         compile_vocabulary_normalisation_factors=False,
                         init_test_data=False)
    profile_from_batch = 0
    start_batch = 0
    stop_batch = start_batch + batches
    if args.start_on_first_epoch_end:
        batches_per_epoch = int(ceil(model._datasets.train_data_length / hparams.get('batch_size')))
        profile_from_batch = batches_per_epoch
        stop_batch = batches_per_epoch + batches
    pbar = ProgressBar(max_value=stop_batch)
    pbar.start()
    workdir = os.path.join(model._workdir, 'profiler')
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    profiler = TfProfiler(logdir=workdir)
    dataset = model._datasets.dataset_train.__iter__()
    for batch in range(start_batch, stop_batch):
        if args.include_dataset_init or batch > profile_from_batch:
            print(f'Profiling batch {batch}')
            profiler.start_if_not_running()
            with Trace('batch', step_num=batch, _r=1):
                _ = next(dataset)
        else:
            _ = next(dataset)  # Just discard data, profiling will happen some other time
        pbar.update(batch)
    profiler.stop()
    pbar.finish()

subparser.set_defaults(func=profile_data_pipeline)

args = parser.parse_args()
args.func(args)  # Callback to trigger func with args
