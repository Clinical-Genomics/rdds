import os.path
import pandas as pd
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2
from google.protobuf import text_format
import tensorflow as tf
import tensorflow_gnn as tfgnn

from .. import WORKDIR, _LOGGER
from .schema import _SCHEMA, _DUMMY_DATA

_HPO_PHEN_TO_GENE_TSV = '/rdds/tmp/dataset-hpo/phenotype_to_genes.txt'
_HPO_GENES_TO_DISEASE = '/rdds/tmp/dataset-hpo/genes_to_disease.txt'
_HPO_FREQUENCY_TO_DISEASE = '/rdds/tmp/dataset-hpo/phenotype.hpoa'

class Phen2GenDatasetCompiler:

    def __init__(self,
                 hpo_phen_to_gene_tsv: str = _HPO_PHEN_TO_GENE_TSV,
                 hpo_genes_to_disease: str = _HPO_GENES_TO_DISEASE,
                 hpo_frequency_to_disease: str = _HPO_FREQUENCY_TO_DISEASE,
                 cleartext_schema: str = _SCHEMA,
                 tfrecord_output_path: str = os.path.join(WORKDIR, 'dataset.tfrecord')):
        self._hpo_phen_to_gene_tsv = hpo_phen_to_gene_tsv
        self._hpo_genes_to_disease_tsv = hpo_genes_to_disease
        self._hpo_frequency_to_disease = hpo_frequency_to_disease
        self._output_tfrecord_file_path = tfrecord_output_path
        self._cleartext_schema = cleartext_schema
        self._schema = text_format.Merge(self._cleartext_schema, schema_pb2.GraphSchema())
        _LOGGER.info(f"Schema:\n{self._schema}")
        self._graph_spec = tfgnn.create_graph_spec_from_schema_pb(self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def graph_spec(self):
        return self._graph_spec

    @staticmethod
    def _construct_hpo_nodes(df: pd.DataFrame) -> tfgnn.NodeSet:
        df = df.copy(deep=True)
        # Assumes that all hpo_names are identical to hpo_id
        df = df.drop_duplicates('hpo_id')
        hpo_id = df.hpo_id.map(lambda hpo_str: int(hpo_str.replace('HP:', ''))).values
        hpo_name = df.hpo_name.values
        node_set = tfgnn.NodeSet.from_fields(
            sizes=tf.constant([len(hpo_id)], dtype=tf.int64),
            features={
                "hpo_id": hpo_id,
                "hpo_name": hpo_name
            }
        )
        return node_set

    @staticmethod
    def _construct_gene_nodes(df: pd.DataFrame) -> tfgnn.NodeSet:
        df = df.copy(deep=True)
        df = df.drop_duplicates('ncbi_gene_id')
        gene_id = df.ncbi_gene_id.map(lambda gene_str: int(gene_str.replace('NCBIGene:', ''))).values
        gene_name = df.gene_symbol.values
        node_set = tfgnn.NodeSet.from_fields(
            sizes=tf.constant([len(gene_id)], dtype=tf.int64),
            features={
                "gene_id": gene_id,
                "gene_name": gene_name
            }
        )
        return node_set

    @staticmethod
    def _construct_disease_nodes(df: pd.DataFrame) -> tfgnn.NodeSet:
        # TODO: Lookup disease name (only URLs are present in TSV data)
        df = df.copy(deep=True)
        disease_id = df.disease_id.drop_duplicates().values
        node_set = tfgnn.NodeSet.from_fields(
            sizes=tf.constant([len(disease_id)], dtype=tf.int64),
            features={
                "disease_id": disease_id,
                "disease_name": [''] * len(disease_id)
            }
        )
        return node_set

    def compile(self):
        """ Preprocess input files and write examples of Graph to TFRecord file """
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

        # Prepare OMIM HPO Frequency to Disease mappings
        df_frequency_to_disease = pd.read_csv(self._hpo_frequency_to_disease,
                                              header=4,
                                              on_bad_lines=on_bad_lines,
                                              delimiter='\t')
        ## Drop all not of datatype 'aspect:phenotypic abnormality', https://hpo.jax.org/browse/term/HP:0000118
        df_frequency_to_disease = df_frequency_to_disease[df_frequency_to_disease.aspect == 'P']
        # TODO: Decode frequency term according to https://obophenotype.github.io/human-phenotype-ontology/annotations/frequency/
        # TODO: Make use of 'qualifier' and NOT annotation for negative associations

        # Define nodes
        hpo_nodes = self._construct_hpo_nodes(df=df_phenotype_to_genes)
        gene_nodes = self._construct_gene_nodes(df=df_genes_to_disease)
        disease_nodes = self._construct_disease_nodes(df=df_genes_to_disease)
        # TODO: Add variants
        # TODO: Add all of NCBI genes (not just disease genes)

        # Define context
        context = None

        return

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
