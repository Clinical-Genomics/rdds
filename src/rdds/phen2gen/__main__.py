import argparse

from . import WORKDIR

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

    case_spec = prepare_clinical_cases(cases_dir=args.patient_case_dir)

    dataset_compiler = Phen2GenDatasetCompiler()
    # dataset_compiler.compile_graph_blob()  # FIXME: prebuilt, cached on disk
    graph_static = dataset_compiler._load_intermediate_graph()

    if False:
        variant_nodes = dataset_compiler._construct_variant_nodes(
            vcf_path=case_spec['popularyak']['vcf']
        )


        with open('/rdds/variant.graph', 'wb') as fp:
            pickle.dump(variant_nodes, fp)

    with open('/rdds/variant.graph', 'rb') as fp:
        variant_nodes = pickle.load(fp)

    variant_gene_edges = Phen2GenDatasetCompiler._construct_variant_gene_edges(
        vcf_path=case_spec['popularyak']['vcf'],
        variant_nodes=variant_nodes,
        gene_nodes=graph_static.gene_nodes
    )

    with open('/rdds/variant-gene-edges.graph', 'wb') as fp:
        pickle.dump(variant_gene_edges, fp)
    print('done')
    exit(0)

    graph_popularyak = dataset_compiler._build_graph(intermediate_graph=graph_static,
                                                     variant_nodes=variant_nodes,
                                                     variant_gene_edges=variant_gene_edges)

subparser.set_defaults(func=lambda args: build_train(args))

args = parser.parse_args()
args.func(args)  # Callback to trigger func with args
