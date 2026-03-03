import os.path
import pandas as pd
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2
from google.protobuf import text_format
import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np
import hpotk
from typing import List, Union, Dict
from dataclasses import dataclass
import pickle
import os

from .. import WORKDIR, _LOGGER
from .schema import _SCHEMA, _DUMMY_DATA
from rdds.lib.checksum import checksum
from rdds.lib.vcf import VCFReader, ParsableVariant

_HPO_PHEN_TO_GENE_TSV = '/rdds/tmp/dataset-hpo/phenotype_to_genes.txt'
_HPO_GENES_TO_DISEASE = '/rdds/tmp/dataset-hpo/genes_to_disease.txt'
_HPO_FREQUENCY_TO_DISEASE = '/rdds/tmp/dataset-hpo/phenotype.hpoa'
_HPO_ONTOLOGY = '/rdds/tmp/dataset-hpo/hp.json'
_HGNC_GENES = '/rdds/tmp/dataset-hgnc/hgnc_complete_set_2026-01-06.txt'



def _lookup_idx(value: Union[str, int],
                arr: np.ndarray,
                allow_misses_with_message: str = None) -> Union[int, None]:
    """
    Helper function to find index of value in arr.
    :param allow_misses_with_message: Allow no match for value in arr, logging a message
    """
    if isinstance(value, str):
        value: bytes = value.encode('utf-8')
    idx, = np.nonzero(arr == value)
    assert isinstance(idx, np.ndarray), idx
    assert len(idx) <= 1, (value, idx, 'ambiguous, multiple matches', arr[idx])
    if len(idx) == 0:
        if allow_misses_with_message:
            _LOGGER.warning(allow_misses_with_message)
            return None
        else:
            raise ValueError(f"value {value} not found in arr, got idx {idx}")
    return idx[0]

def _variant_id(parsed_variant: ParsableVariant) -> str:
    # Create variant ID from ID as well as REF-ALT for examples such as 1_15274_A_T;1_15274_A_G
    return str(parsed_variant.CHROM) + '.' + str(parsed_variant.POS) + '-' + parsed_variant.ID + '-' + parsed_variant.REF + '-' + parsed_variant.ALT


@dataclass
class IntermediateGraph:
    hpo_nodes: tfgnn.NodeSet
    gene_nodes: tfgnn.NodeSet
    disease_nodes: tfgnn.NodeSet
    hpo_hpo_edges: tfgnn.EdgeSet
    gene_hpo_edges: tfgnn.EdgeSet
    disease_hpo_edges: tfgnn.EdgeSet
    gene_disease_edges: tfgnn.EdgeSet


class Phen2GenDatasetCompiler:

    def __init__(self,
                 hpo_phen_to_gene_tsv: str = _HPO_PHEN_TO_GENE_TSV,
                 hpo_genes_to_disease: str = _HPO_GENES_TO_DISEASE,
                 hpo_frequency_to_disease: str = _HPO_FREQUENCY_TO_DISEASE,
                 hpo_ontology: str = _HPO_ONTOLOGY,
                 hgnc_genes: str = _HGNC_GENES,
                 cleartext_schema: str = _SCHEMA,
                 tfrecord_output_path: str = os.path.join(WORKDIR, 'dataset.tfrecord')):
        self._hpo_phen_to_gene_tsv = hpo_phen_to_gene_tsv
        self._hpo_genes_to_disease_tsv = hpo_genes_to_disease
        self._hpo_frequency_to_disease = hpo_frequency_to_disease
        self._hpo_ontology = hpo_ontology
        self._hgnc_genes = hgnc_genes
        self._output_tfrecord_file_path = tfrecord_output_path
        self._cleartext_schema = cleartext_schema
        self._schema = text_format.Merge(self._cleartext_schema, schema_pb2.GraphSchema())
        _LOGGER.info(f"Schema:\n{self._schema}")
        # NOTE: indices_dtype must  go hand in hand with Sampler expected indices dtype
        self._graph_spec = tfgnn.create_graph_spec_from_schema_pb(self._schema, indices_dtype=tf.int64)
        self._intermediate_graph_storage_location = os.path.join(WORKDIR, 'graph-data/intermediate-graph.blob')

    @property
    def schema(self) -> schema_pb2.GraphSchema:
        return self._schema

    @property
    def graph_spec(self) -> tfgnn.GraphTensorSpec:
        return self._graph_spec

    @staticmethod
    def _construct_hpo_nodes(hpo_ontology: hpotk.Ontology) -> tfgnn.NodeSet:
        _LOGGER.info("Constructing HPO nodes")
        nodeset_hpo_id: List[int] = []
        nodeset_hpo_id_full: List[str] = []
        nodeset_hpo_name: List[str] = []

        for hpo_term in hpo_ontology.terms:
            _LOGGER.debug(f"Adding node for HPO term {hpo_term}")
            hpo_id_str: str = hpo_term.identifier.value
            assert isinstance(hpo_id_str, str)
            assert len(hpo_id_str) > 0 and 'HP:' in hpo_id_str
            hpo_id_int: int = int(hpo_id_str.replace('HP:', ''))
            assert hpo_id_int > 0
            hpo_name: str = hpo_term.name  # Short name
            if hpo_term.definition:
                hpo_definition: str = hpo_term.definition.definition  # A more descriptive explanation of the term
            else:
                hpo_definition: str = ''
            # Add to the node set
            nodeset_hpo_id.append(hpo_id_int)
            nodeset_hpo_id_full.append(hpo_id_str)
            nodeset_hpo_name.append(hpo_name)
        node_set = tfgnn.NodeSet.from_fields(
            sizes=tf.constant([len(nodeset_hpo_id)], dtype=tf.int64),
            features={
                "#id": tf.constant(nodeset_hpo_id, dtype=tf.int64),
                "hpo_id": tf.constant(nodeset_hpo_id, dtype=tf.int64),  # TODO: Deprecate, replaced by #id
                "hpo_id_full": tf.constant(nodeset_hpo_id_full, dtype=tf.string),
                "hpo_name": tf.constant(nodeset_hpo_name, dtype=tf.string)
            }
        )
        _LOGGER.info(f"Added nodes for {node_set.total_size} HPO terms")
        return node_set

    @staticmethod
    def _construct_gene_nodes(df: pd.DataFrame) -> tfgnn.NodeSet:
        df = df.copy(deep=True)
        gene_id = df.hgnc_id.map(lambda gene_str: int(gene_str.replace('HGNC:', ''))).values  # HGNC:32
        gene_symbol = df.symbol.values  # CARD9
        node_set = tfgnn.NodeSet.from_fields(
            sizes=tf.constant([len(gene_id)], dtype=tf.int64),
            features={
                "#id": tf.constant(df.index.values, dtype=tf.int64),
                "gene_id": tf.constant(gene_id, dtype=tf.int64),
                "gene_symbol": tf.constant(gene_symbol, dtype=tf.string)
            }
        )
        _LOGGER.info(f"Added nodes for {node_set.total_size} genes")
        return node_set

    @staticmethod
    def _construct_disease_nodes(df_gene_to_disease: pd.DataFrame,
                                 df_frequency_to_disease: pd.DataFrame) -> tfgnn.NodeSet:
        """
        Construct nodes from OMIM and ORPHANET disease IDs
        """
        # TODO: Download complete set of OMIM, ORPHANET disease IDs from source, input files here might be incomplete!
        # TODO: Lookup disease name (only URLs are present in TSV data)
        _LOGGER.info("Constructing disease nodes")

        gene_disease_ids = df_gene_to_disease.disease_id.copy()
        frequency_to_disease_ids = df_frequency_to_disease.database_id.copy()
        disease_ids = pd.concat((gene_disease_ids, frequency_to_disease_ids), axis=0, ignore_index=True)
        disease_ids = disease_ids.drop_duplicates()
        disease_ids = disease_ids.values
        node_set = tfgnn.NodeSet.from_fields(
            sizes=tf.constant([len(disease_ids)], dtype=tf.int64),
            features={
                "#id": tf.constant(range(0, len(disease_ids)), dtype=tf.int64),
                "disease_id": tf.constant(disease_ids, dtype=tf.string),
                "disease_name": tf.constant([''] * len(disease_ids), dtype=tf.string)
            }
        )
        _LOGGER.info(f"Added nodes for {node_set.total_size} diseases")
        return node_set

    @staticmethod
    def _construct_hpo_hpo_edges(hpo_ontology: hpotk.Ontology, hpo_node_set: tfgnn.NodeSet) -> tfgnn.EdgeSet:
        """
        https://github.com/ielis/hpo-toolkit/blob/main/docs/user-guide/use-hierarchy.rst#hierarchy-traversals
        """
        edgeset_sources = []
        edgeset_targets = []
        hpo_ids: np.ndarray[bytes] = hpo_node_set.features['hpo_id_full'].numpy()

        def _lookup_idx(hpo_id: str) -> int:
            hpo_id: bytes = hpo_id.encode('utf-8')
            idx, = np.nonzero(hpo_ids == hpo_id)
            assert isinstance(idx, np.ndarray), idx
            assert len(idx) == 1, (hpo_id, idx, 'not found')
            return idx[0]

        for index, hpo_id_bytes in enumerate(hpo_ids):
            _LOGGER.debug(f"Mapping HPO node ID: {hpo_id_bytes}")
            hpo_id_bytes: bytes
            hpo_id_str: str = hpo_id_bytes.decode('utf-8')
            hpo_id_in_onthology = hpo_ontology.get_term(term_id=hpo_id_str)
            assert isinstance(hpo_id_in_onthology, hpotk.model._term.DefaultTerm), (hpo_id_str, hpo_id_in_onthology)

            # Define edges for HPO term parents
            for parent in hpo_ontology.graph.get_parents(source=hpo_id_str):
                _LOGGER.debug(f"{parent.value} < {hpo_id_str}")
                parent_idx = _lookup_idx(parent.value)
                edgeset_sources.append(index)
                edgeset_targets.append(parent_idx)

            # Define edges for HPO term children
            for children in hpo_ontology.graph.get_children(source=hpo_id_str):
                _LOGGER.debug(f"{hpo_id_str} > {children.value}")
                children_idx = _lookup_idx(children.value)
                edgeset_sources.append(index)
                edgeset_targets.append(children_idx)

        edge_set = tfgnn.EdgeSet.from_fields(
            sizes=tf.constant([len(edgeset_sources)]),
            adjacency=tfgnn.Adjacency.from_indices(
                source=("hpo", tf.constant(edgeset_sources, dtype=tf.int64)),
                target=("hpo", tf.constant(edgeset_targets, dtype=tf.int64))
            )
        )
        _LOGGER.info(f"Added {edge_set.total_size} edges for hpo-hpo terms")
        return edge_set

    @staticmethod
    def _construct_gene_hpo_edges(phenotype_to_genes: pd.DataFrame,
                                  hpo_nodes: tfgnn.NodeSet,
                                  gene_nodes: tfgnn.NodeSet,
                                  ) -> tfgnn.EdgeSet:
        """
        Construct HPO to Disease-associated-gene edge.
        Consists of both OMIM and ORPHANET associations in a hpo-to-gene 1-by-1 mapping.
        There are multiple hpo->gene mappings per HPO term, so there are multiple edges per hpo term
        to gene.

        TODO: Select OMIM or ORPHANET as hpo-gene association source, see phenotype_to_genes.disease_id
        """
        phenotype_to_genes = phenotype_to_genes.copy(deep=True)

        _LOGGER.info("Creating edges hpo-to-gene")
        hpo_ids: np.ndarray = hpo_nodes.features["hpo_id_full"].numpy()  # str
        gene_symbols: np.ndarray = gene_nodes.features["gene_symbol"].numpy()  # str

        hpo_id_idx = phenotype_to_genes.hpo_id.map(lambda hpo_id: _lookup_idx(hpo_id, arr=hpo_ids))
        gene_symbol_idx = phenotype_to_genes.gene_symbol.map(lambda gene_symbol: _lookup_idx(gene_symbol, arr=gene_symbols))

        phenotype_to_genes['nodeset_hpo_idx'] = hpo_id_idx
        phenotype_to_genes['nodeset_gene_idx'] = gene_symbol_idx

        edge_set = tfgnn.EdgeSet.from_fields(
            sizes=tf.constant([len(phenotype_to_genes)]),
            adjacency=tfgnn.Adjacency.from_indices(
                source=("gene", tf.constant(phenotype_to_genes.nodeset_gene_idx.values, dtype=tf.int64)),
                target=("hpo", tf.constant(phenotype_to_genes.nodeset_hpo_idx.values, dtype=tf.int64))
            )
        )
        _LOGGER.info(f"Added {edge_set.total_size} edges for gene-hpo terms")
        return edge_set

    @staticmethod
    def _construct_hpo_disease_edges(hpo_frequency_to_disease: pd.DataFrame,
                                     hpo_nodes: tfgnn.NodeSet,
                                     disease_nodes: tfgnn.NodeSet) -> tfgnn.EdgeSet:
        """
        TODO: Parse the .qualifier field to make use of NOT negative association
        TODO: Parse frequency to provide "support" feature, indicating number of patients per disease
        TODO: Parse frequency to provide a normalisation constant
        """
        _LOGGER.info('Creating edges hpo-to-gene')

        hpo_frequency_to_disease = hpo_frequency_to_disease.copy(deep=True)

        from fractions import Fraction
        def _parse_frequency_field(value: Union[str, None]) -> float:
            """
            Examples: 1/4, HP:0011461, 32% or NaN

            Excluded        HP:0040285 	0% of affected individuals
            Very rare       HP:0040284 	1–4% of affected individuals
            Occasional      HP:0040283 	5–29% of affected individuals
            Frequent        HP:0040282 	30–79% of affected individuals
            Very frequent   HP:0040281 	80–99% of affected individuals
            Obligate        HP:0040280  100% of affected individuals

            Source: https://obophenotype.github.io/human-phenotype-ontology/annotations/frequency/
            """
            lookup_table = {  # As an average of the bounds in the above specification, (0, 1)
                'HP:0040285': 0.0,  # 0% of affected individuals
                'HP:0040284': (1 + 4) / 2.0 / 100.0,  # 1–4% of affected individuals
                'HP:0040283': (5 + 29) / 2.0 / 100.0,  # 5–29% of affected individuals
                'HP:0040282': (30 + 79) / 2.0 / 100.0,  # 30–79% of affected individuals
                'HP:0040281': (80 + 99) / 2.0 / 100.0,  # 80–99% of affected individuals
                'HP:0040280': 1.0  # 100% of affected individuals
            }
            if not (value == value):  # nan check
                return None
            if 'HP' in value:
                return lookup_table[value]
            if '%' in value:
                return float(value.replace('%','')) / 100.0
            fraction = Fraction(value)
            frequency = fraction.numerator / fraction.denominator
            return frequency

        hpo_frequency_to_disease['frequency_parsed'] = hpo_frequency_to_disease.frequency.map(lambda frq_value: _parse_frequency_field(frq_value))
        node_disease: np.ndarray = disease_nodes.features['disease_id'].numpy()  # str
        node_hpo: np.ndarray = hpo_nodes.features['hpo_id_full'].numpy()  # str
        hpo_frequency_to_disease['node_hpo_idx'] = hpo_frequency_to_disease.hpo_id.map(lambda hpo_id_str: _lookup_idx(value=hpo_id_str, arr=node_hpo))
        hpo_frequency_to_disease['node_disease_idx'] = hpo_frequency_to_disease.database_id.map(lambda database_id_str: _lookup_idx(value=database_id_str, arr=node_disease))
        # TODO: Add frequency_parsed as feature to edge
        edge_set = tfgnn.EdgeSet.from_fields(
            sizes=tf.constant([len(hpo_frequency_to_disease)]),
            adjacency=tfgnn.Adjacency.from_indices(
                source=("hpo", tf.constant(hpo_frequency_to_disease.node_hpo_idx.values, dtype=tf.int64)),
                target=("disease", tf.constant(hpo_frequency_to_disease.node_disease_idx.values, dtype=tf.int64))
            )
        )
        _LOGGER.info(f"Added {edge_set.total_size} edges for hpo-disease terms")
        return edge_set

    @staticmethod
    def _construct_gene_disease_edges(genes_to_disease: pd.DataFrame,
                                      gene_nodes: tfgnn.NodeSet,
                                      disease_nodes: tfgnn.NodeSet) -> tfgnn.EdgeSet:
        _LOGGER.info("Constructing gene-disease edges")
        genes_to_disease = genes_to_disease.copy(deep=True)
        genes_to_disease = genes_to_disease.query("gene_symbol!='-'")  # Drop entries where no gene symbol associated
        gene_symbols = gene_nodes.features['gene_symbol'].numpy()  # str, BRCA1
        disease_ids = disease_nodes.features['disease_id'].numpy()  # str, OMIM: or ORPHANET:
        gene_symbol_idx = genes_to_disease.gene_symbol.map(lambda gene_symbol: _lookup_idx(gene_symbol, arr=gene_symbols))
        disease_id_idx = genes_to_disease.disease_id.map(lambda disease_id: _lookup_idx(disease_id, arr=disease_ids))
        edge_set = tfgnn.EdgeSet.from_fields(
            sizes=tf.constant([len(genes_to_disease)]),
            adjacency=tfgnn.Adjacency.from_indices(
                source=("gene", tf.constant(gene_symbol_idx.values, dtype=tf.int64)),
                target=("disease", tf.constant(disease_id_idx.values, dtype=tf.int64))
            )
        )
        _LOGGER.info(f"Added {edge_set.total_size} edges for gene->disease terms")
        return edge_set

    @staticmethod
    def _construct_variant_nodes(vcf_path: str) -> tfgnn.NodeSet:
        _LOGGER.info("Constructing variant nodes")
        vcf_reader = VCFReader(fname=vcf_path)
        variant_ids: List[str] = []
        genmod_rank_scores: List[float] = []
        for variant in vcf_reader:
            variant_parsed = ParsableVariant(variant=variant, vep_csq_description=vcf_reader.csq_description)
            variant_ids.append(_variant_id(parsed_variant=variant_parsed))
            genmod_rank_scores.append(variant_parsed.RankScore_value)
        node_set = tfgnn.NodeSet.from_fields(
            sizes=tf.constant([len(variant_ids)], dtype=tf.int64),
            features={
                "#id":  tf.range(0, len(variant_ids), dtype=tf.int64),
                "variant_id": tf.constant(variant_ids, dtype=tf.string),
                "genmod_rank_score": tf.constant(genmod_rank_scores, dtype=tf.float32),
                "label": tf.constant([0] * len(variant_ids), dtype=tf.float32)
            }
        )
        _LOGGER.info(f"Added {node_set.total_size} variant nodes")
        return node_set

    @staticmethod
    def _find_variant_gene_edge(variant_index_start: int,
                                variant_index_end: int,
                                vcf_path: str,
                                variant_node_ids: np.ndarray,
                                gene_node_ids: np.ndarray,
                                result_queue,
                                tmp_dir_path: str):
        """
        Helper multiprocessing method to shard variants and lookup related gene idx.
        """
        from rdds.lib.process_pool import MULTIPROCESSING_LOGGER
        vcf_reader = VCFReader(fname=vcf_path, unpack_if_gzipped=False)  # Don't unpack to RAM (risk of OoM)
        variants = list(vcf_reader)
        MULTIPROCESSING_LOGGER.info(f"Loaded {len(variants)} variants")
        csq_description = vcf_reader.csq_description
        del vcf_reader
        variants = variants[variant_index_start:variant_index_end+1]  # Account for indexing not inclusive
        MULTIPROCESSING_LOGGER.info(f"Will process {len(variants)} variants")
        variant_ids = []
        gene_ids = []
        for variant in variants:
            try:
                variant_parsed = ParsableVariant(variant=variant, vep_csq_description=csq_description)
                variant_vcf_id = variant_parsed.ID
                gene_id = int(variant_parsed.CSQ_HGNC_ID)
                gene_symbol = variant_parsed.CSQ_SYMBOL
                if gene_id is None:
                    MULTIPROCESSING_LOGGER.warning(f"Variant-Gene edger: Ignoring variant {variant_vcf_id} due to no annotated gene_symbol: '{gene_symbol}'")
                    continue
                # TODO: FIXME: lookup will fail for variants due to some GRCh37 specific gene names as well as RNA specific annotations
                gene_idx = _lookup_idx(value=gene_id,
                                       arr=gene_node_ids,
                                       allow_misses_with_message = f"Variant {variant_vcf_id}, gene={gene_symbol}, {gene_id} not found in node genes")
                if gene_idx:
                    variant_idx = _lookup_idx(value=_variant_id(parsed_variant=variant_parsed), arr=variant_node_ids)
                    variant_ids.append(variant_idx)
                    gene_ids.append(gene_idx)
                else:
                    raise ValueError(f"Found no variant-gene-link between {variant_vcf_id} {gene_id} {gene_symbol}")
            except Exception as e:
                if variant.CHROM.lower() == 'mt':
                    # TODO: Input data should be annotated with MT HGNC IDs
                    MULTIPROCESSING_LOGGER.warning(f"Error parsing variant {vcf_path, variant.CHROM, variant.POS, variant.ID}: {e}")
                    MULTIPROCESSING_LOGGER.warning("MT SNVs not supported (missing HGNC annotations) - continuing")
                    continue
                MULTIPROCESSING_LOGGER.error(f"Error parsing variant {vcf_path, variant.CHROM, variant.POS, variant.ID}: {e}")
                raise e
        result_dict = {
            'variant_index_start': variant_index_start,
            'variant_ids': variant_ids,
            'gene_ids': gene_ids
        }
        output_file_name = os.path.join(tmp_dir_path, str(variant_index_start))
        MULTIPROCESSING_LOGGER.info(f"Storing node-gene mapping in {output_file_name}")
        with open(output_file_name, 'wb') as fp:
            pickle.dump(result_dict, fp)
        result_queue.put(output_file_name)

    @staticmethod
    def _construct_variant_gene_edges(vcf_path: str,
                                      variant_nodes: tfgnn.NodeSet,
                                      gene_nodes: tfgnn.NodeSet,
                                      n_workers: int = os.cpu_count()) -> tfgnn.EdgeSet:
        """
        Construct variant-gene edges.

        This is a very costly method in terms of compute, since every variant needs
        to be matched variant <-> gene index by lookup tables.

        Store intermediate results as pickled objects on disk,
        as they are too big to be passed in pipes, queues.
        """
        from rdds.lib.process_pool import ProcessPool
        from tempfile import TemporaryDirectory
        import os
        n_workers = min(n_workers, os.cpu_count())
        _LOGGER.info(f"Will use {n_workers} workers")
        variant_node_ids = variant_nodes.features['variant_id'].numpy()
        gene_node_ids = gene_nodes.features['gene_id'].numpy()
        vcf_reader = VCFReader(fname=vcf_path)
        number_of_variants = vcf_reader.number_of_variants
        del vcf_reader
        variant_indices = np.arange(number_of_variants)
        variant_indices_jobs = np.array_split(variant_indices, indices_or_sections=n_workers)
        job_kwargs = []
        result_queue = ProcessPool.get_context().SimpleQueue()
        tmp_dir = TemporaryDirectory(dir=WORKDIR, prefix=os.path.basename(vcf_path)+'-')
        tmp_dir_path = tmp_dir.name
        _LOGGER.info(f"Temporary work dir: {tmp_dir_path}")
        for variant_index_array in variant_indices_jobs:
            variant_index_array = list(variant_index_array)
            job_kwargs.append({'variant_index_start': variant_index_array[0],
                               'variant_index_end': variant_index_array[-1],
                               'vcf_path': vcf_path,
                               'variant_node_ids': variant_node_ids,
                               'gene_node_ids': gene_node_ids,
                               'result_queue': result_queue,
                               'tmp_dir_path': tmp_dir_path})
        pool = ProcessPool(function=Phen2GenDatasetCompiler._find_variant_gene_edge,
                           kwargs=job_kwargs,
                           workers=n_workers)
        completed_tasks = pool.run()

        # Store pickled object paths from workers
        results_pickled = []
        for task in completed_tasks:
            assert task.process.exitcode == 0, task
            results_pickled.append(result_queue.get())
            if len(results_pickled) == n_workers:
                _LOGGER.info('Recieved all worker results')
                break
            else:
                _LOGGER.debug(f"Recieved result: {results_pickled[-1]}")
        pool.close()
        del pool

        # Load pickled arrays
        results = []
        for path in results_pickled:
            with open(path, 'rb') as fp:
                results.append(pickle.load(fp))
        results = sorted(results, key=lambda d: d['variant_index_start'])

        # Load lists of variant and gene idx
        gene_idx = []
        variant_idx = []
        for result in results:
            variant_idx += result['variant_ids']
            gene_idx += result['gene_ids']

        edge_set = tfgnn.EdgeSet.from_fields(
            sizes=tf.constant([len(variant_idx)]),
            adjacency=tfgnn.Adjacency.from_indices(
                source=("variant", tf.constant(variant_idx, dtype=tf.int64)),
                target=("gene", tf.constant(gene_idx, dtype=tf.int64))
            )
        )
        _LOGGER.info(f"Added {edge_set.total_size} variant->gene edges")
        return edge_set


    def compile_graph_blob(self):
        """ Preprocess input files of non-patient case specific origin and write to intermediate IntermediateGraph blob """
        on_bad_lines = "error"
        # Prepare HPO Phenotype to Genotype TSV
        df_phenotype_to_genes = pd.read_csv(self._hpo_phen_to_gene_tsv,
                                            low_memory=False,
                                            on_bad_lines=on_bad_lines,
                                            delimiter='\t')

        # Prepare HPO phenotype ID to NCBI gene ID
        df_genes_to_disease = pd.read_csv(self._hpo_genes_to_disease_tsv,
                                          low_memory=False,
                                          on_bad_lines=on_bad_lines,
                                          delimiter='\t')

        # Prepare OMIM, ORPHANET HPO Frequency to Disease mappings
        df_frequency_to_disease = pd.read_csv(self._hpo_frequency_to_disease,
                                              header=4,
                                              on_bad_lines=on_bad_lines,
                                              delimiter='\t')
        ## Drop all not of datatype 'aspect:phenotypic abnormality', https://hpo.jax.org/browse/term/HP:0000118
        df_frequency_to_disease = df_frequency_to_disease[df_frequency_to_disease.aspect == 'P']

        hpo_ontology: hpotk.Ontology = hpotk.load_ontology(self._hpo_ontology)

        # Prepare HGNC genes
        df_hgnc_genes = pd.read_csv(self._hgnc_genes,
                                    low_memory=False,
                                    on_bad_lines=on_bad_lines,
                                    delimiter='\t')

        # Define nodes
        hpo_nodes = self._construct_hpo_nodes(hpo_ontology=hpo_ontology)
        gene_nodes = self._construct_gene_nodes(df=df_hgnc_genes)
        disease_nodes = self._construct_disease_nodes(df_gene_to_disease=df_genes_to_disease,
                                                      df_frequency_to_disease=df_frequency_to_disease)

        # Define edges
        hpo_hpo_edges = self._construct_hpo_hpo_edges(hpo_ontology=hpo_ontology, hpo_node_set=hpo_nodes)
        gene_hpo_edges = self._construct_gene_hpo_edges(phenotype_to_genes=df_phenotype_to_genes,
                                                        gene_nodes=gene_nodes,
                                                        hpo_nodes=hpo_nodes)
        disease_hpo_edges = self._construct_hpo_disease_edges(hpo_frequency_to_disease=df_frequency_to_disease,
                                                              hpo_nodes=hpo_nodes,
                                                              disease_nodes=disease_nodes)
        gene_disease_edges = self._construct_gene_disease_edges(genes_to_disease=df_genes_to_disease,
                                                               gene_nodes=gene_nodes,
                                                               disease_nodes=disease_nodes)

        intermediate_graph = IntermediateGraph(hpo_nodes=hpo_nodes,
                                               gene_nodes=gene_nodes,
                                               disease_nodes=disease_nodes,
                                               hpo_hpo_edges=hpo_hpo_edges,
                                               gene_hpo_edges=gene_hpo_edges,
                                               disease_hpo_edges=disease_hpo_edges,
                                               gene_disease_edges=gene_disease_edges)
        self._store_intermediate_graph(intermediate_graph)

    def _store_intermediate_graph(self, intermediate_graph: IntermediateGraph):
        """
        Save the non-case specific graph data to disk for reuse
        """
        try:
            os.mkdir(os.path.dirname(self._intermediate_graph_storage_location))
        except FileExistsError:
            pass
        with open(self._intermediate_graph_storage_location, mode='wb') as fp:
            pickle.dump(intermediate_graph, fp)
        graph_checksum = checksum(file_path=self._intermediate_graph_storage_location, algorithm='sha256')
        _LOGGER.info(f"Wrote {self._intermediate_graph_storage_location}:{graph_checksum}")
        with open(self._intermediate_graph_storage_location + '.sha256', 'w') as fp:
            fp.write(f"{graph_checksum} {self._intermediate_graph_storage_location}")

    def _load_intermediate_graph(self, intermediate_graph_storage_location: str = None) -> IntermediateGraph:
        """
        Load IntermediateGraph from disk
        """
        if intermediate_graph_storage_location is None:
            intermediate_graph_storage_location = self._intermediate_graph_storage_location
        _LOGGER.info(f"Loading intermediate graph from blob: {intermediate_graph_storage_location}")
        with open(intermediate_graph_storage_location + '.sha256', 'r') as fp:
            expected_graph_checksum = fp.read().split(' ')[0]
        graph_checksum = checksum(file_path=intermediate_graph_storage_location, algorithm='sha256')
        assert graph_checksum == expected_graph_checksum, (graph_checksum, expected_graph_checksum)
        with open(self._intermediate_graph_storage_location, 'rb') as fp:
            intermediate_graph: IntermediateGraph = pickle.load(fp)
        return intermediate_graph

    @staticmethod
    def _build_graph(intermediate_graph: IntermediateGraph,
                     variant_nodes: tfgnn.NodeSet,
                     variant_gene_edges: tfgnn.EdgeSet) -> tfgnn.GraphTensor:

        node_sets = {
            'hpo': intermediate_graph.hpo_nodes,
            'disease': intermediate_graph.disease_nodes,
            'gene': intermediate_graph.gene_nodes,
            'variant': variant_nodes
        }

        edge_sets = {
            'hpo>hpo': intermediate_graph.hpo_hpo_edges,
            'gene>hpo': intermediate_graph.gene_hpo_edges,
            'hpo>disease': intermediate_graph.disease_hpo_edges,
            'gene>disease': intermediate_graph.gene_disease_edges,
            'variant>gene': variant_gene_edges
        }

        graph = tfgnn.GraphTensor.from_pieces(
            node_sets=node_sets,
            edge_sets=edge_sets
        )

        return graph

    def _yield_graph_tensor(self,
                            dummy_data: bool = False,
                            n_dummy_samples: int = 10) -> tfgnn.GraphTensor:
        """
        :param dummy_data: Yield dummy data (for testing)
        """
        raise DeprecationWarning()
        # TODO: Replace with iterator on top of compiled dataset file
        if dummy_data:
            for _ in range(0, n_dummy_samples):
                graph_tensor = tfgnn.random_graph_tensor(self._graph_spec,
                                                         sample_dict=_DUMMY_DATA,
                                                         num_components_range=(1, 2)  # [1, 2)
                                                         )
                yield graph_tensor
            return

        # TODO: Replace with tfgnn.GraphTensor.from_pieces()
        raise NotImplementedError()

    def _write_tfrecord(self, dummy_data: bool = False):
        _LOGGER.info(f"Writing tfrecord file to {self._output_tfrecord_file_path}")
        written_records = 0
        with tf.io.TFRecordWriter(self._output_tfrecord_file_path) as writer:
            for graph in self._yield_graph_tensor(dummy_data=dummy_data):
                example = tfgnn.write_example(graph)
                writer.write(example.SerializeToString())
                written_records += 1
        if written_records == 0:
            raise ValueError("Wrote no records to file")
        _LOGGER.info(f"Wrote {written_records} records")

    @property
    def dataset(self):
        # TODO: Create tf.Dataset instance from TFRecord file
        return None
