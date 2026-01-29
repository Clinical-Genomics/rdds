import os.path
import pandas as pd
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2
from google.protobuf import text_format
import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np
import hpotk
from typing import List, Union
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


def _lookup_idx(value: Union[str, int], arr: np.ndarray) -> int:
    """
    Helper function to find index of value in arr.
    """
    if isinstance(value, str):
        value: bytes = value.encode('utf-8')
    idx, = np.nonzero(arr == value)
    assert isinstance(idx, np.ndarray), idx
    assert len(idx) == 1, (value, idx, 'not found')
    return idx[0]


@dataclass
class IntermediateGraph:
    hpo_nodes: tfgnn.NodeSet
    gene_nodes: tfgnn.NodeSet
    disease_nodes: tfgnn.NodeSet
    hpo_hpo_edges: tfgnn.EdgeSet
    hpo_gene_edges: tfgnn.EdgeSet
    disease_hpo_edges: tfgnn.EdgeSet


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
        self._graph_spec = tfgnn.create_graph_spec_from_schema_pb(self._schema)
        self._intermediate_graph_storage_location = os.path.join(WORKDIR, 'graph-data/intermediate-graph.blob')

    @property
    def schema(self):
        return self._schema

    @property
    def graph_spec(self):
        return self._graph_spec

    @staticmethod
    def _construct_hpo_nodes(hpo_ontology: hpotk.Ontology) -> tfgnn.NodeSet:

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
                "hpo_id": nodeset_hpo_id,
                "hpo_id_full": nodeset_hpo_id_full,
                "hpo_name": nodeset_hpo_name
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
                "gene_id": gene_id,
                "gene_symbol": gene_symbol
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

        gene_disease_ids = df_gene_to_disease.disease_id.copy()
        frequency_to_disease_ids = df_frequency_to_disease.database_id.copy()
        disease_ids = pd.concat((gene_disease_ids, frequency_to_disease_ids), axis=0, ignore_index=True)
        disease_ids = disease_ids.drop_duplicates()
        disease_ids = disease_ids.values
        node_set = tfgnn.NodeSet.from_fields(
            sizes=tf.constant([len(disease_ids)], dtype=tf.int64),
            features={
                "disease_id": disease_ids,
                "disease_name": [''] * len(disease_ids)
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
                source=("hpo", edgeset_sources),
                target=("hpo", edgeset_targets)
            )
        )
        _LOGGER.info(f"Added {edge_set.total_size} edges for hpo-hpo terms")
        return edge_set

    @staticmethod
    def _construct_hpo_gene_edges(phenotype_to_genes: pd.DataFrame,
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
                source=("hpo", phenotype_to_genes.nodeset_hpo_idx.values),
                target=("gene", phenotype_to_genes.nodeset_gene_idx.values)
            )
        )
        _LOGGER.info(f"Added {edge_set.total_size} edges for hpo-gene terms")
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
        edge_set = tfgnn.EdgeSet.from_fields(
            sizes=tf.constant([len(hpo_frequency_to_disease)]),
            adjacency=tfgnn.Adjacency.from_indices(
                source=("hpo", hpo_frequency_to_disease.node_hpo_idx.values),
                target=("disease", hpo_frequency_to_disease.node_disease_idx.values)
            )
        )
        _LOGGER.info(f"Added {edge_set.total_size} edges for hpo-disease terms")
        return edge_set

    @staticmethod
    def _construct_variant_nodes(vcf_path: str) -> tfgnn.NodeSet:
        vcf_reader = VCFReader(fname=vcf_path)
        variant_ids: List[str] = []
        genmod_rank_scores: List[float] = []
        for variant in vcf_reader:
            variant_parsed = ParsableVariant(variant=variant, vep_csq_description=vcf_reader.csq_description)
            variant_ids.append(variant_parsed.ID)
            genmod_rank_scores.append(variant_parsed.RankScore_value)
        node_set = tfgnn.NodeSet.from_fields(
            sizes=tf.constant([len(variant_ids)], dtype=tf.int64),
            features={
                "variant_id": variant_ids,
                "genmod_rank_score": genmod_rank_scores,
                "label": [0] * len(variant_ids)
            }
        )
        return node_set

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
        hpo_gene_edges = self._construct_hpo_gene_edges(phenotype_to_genes=df_phenotype_to_genes,
                                                        gene_nodes=gene_nodes,
                                                        hpo_nodes=hpo_nodes)
        disease_hpo_edges = self._construct_hpo_disease_edges(hpo_frequency_to_disease=df_frequency_to_disease,
                                                              hpo_nodes=hpo_nodes,
                                                              disease_nodes=disease_nodes)

        intermediate_graph = IntermediateGraph(hpo_nodes=hpo_nodes,
                                               gene_nodes=gene_nodes,
                                               disease_nodes=disease_nodes,
                                               hpo_hpo_edges=hpo_hpo_edges,
                                               hpo_gene_edges=hpo_gene_edges,
                                               disease_hpo_edges=disease_hpo_edges)
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

    def _yield_graph_tensor(self,
                            dummy_data: bool = False,
                            n_dummy_samples: int = 10) -> tfgnn.GraphTensor:
        """
        :param dummy_data: Yield dummy data (for testing)
        """
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
