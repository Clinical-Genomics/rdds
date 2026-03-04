import argparse

from . import WORKDIR, _LOGGER

parser = argparse.ArgumentParser(prog='Phen2Gen Model',
                                 description='Data management, model training and inference CLI')
subparsers = parser.add_subparsers()
subparser = subparsers.add_parser('download-prerequisites', help='Download prerequisite data for building dataset')
def _download_prerequisites(args):
    from rdds.dataset_hpo.hpo import HPO
    from rdds.dataset_hgnc.hgnc import HGNC
    hpo = HPO()
    hpo.download()
    hgnc = HGNC()
    hgnc.download()
subparser.set_defaults(func=lambda args: _download_prerequisites(args))

subparser = subparsers.add_parser('precompile-static-dataset', help='Build static part of dataset from upstream data')
def _precompile(args):
    # Build intermediate graph (without variants)
    from rdds.phen2gen.dataset.dataset import Phen2GenDatasetCompiler
    dataset_compiler = Phen2GenDatasetCompiler()
    dataset_compiler.compile_graph_blob()
subparser.set_defaults(func=lambda args: _precompile(args))

subparser = subparsers.add_parser('build-dataset', help='Build dataset for training')
subparser.add_argument('--patient-case-dir', default='/rdds/phenodata', help='Path to VCF file')  # FIXME: default
def build_train(args):
    from .dataset import prepare_clinical_cases
    from .dataset import Phen2GenDatasetCompiler
    import pickle

    case_specs = prepare_clinical_cases(cases_dir=args.patient_case_dir)
    dataset_compiler = Phen2GenDatasetCompiler()
    dataset_compiler.compile_graph_blob()  # FIXME: prebuilt, cached on disk
    for case_name, case_spec in case_specs.items():
        _LOGGER.info(f"Building graph for case: {case_name}")
        dataset_compiler = Phen2GenDatasetCompiler()
        graph_static = dataset_compiler._load_intermediate_graph()
        variant_nodes = dataset_compiler._construct_variant_nodes(vcf_path=case_spec[case_name]['vcf'])
        variant_gene_edges = Phen2GenDatasetCompiler._construct_variant_gene_edges_parallel(
        vcf_path=case_spec[case_name]['vcf'],
        variant_nodes=variant_nodes,
        gene_nodes=graph_static.gene_nodes)
        case_graph = dataset_compiler._build_graph(intermediate_graph=graph_static,
                                                   variant_nodes=variant_nodes,
                                                   variant_gene_edges=variant_gene_edges)
        storage_path = WORKDIR + f"/{case_name}.graph"
        _LOGGER.info(f"Storing graph at {storage_path}")
        with open(storage_path, 'wb') as fp:
            pickle.dump(case_graph, fp)

subparser.set_defaults(func=lambda args: build_train(args))

args = parser.parse_args()
args.func(args)  # Callback to trigger func with args
