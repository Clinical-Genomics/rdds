from glob import glob
import os
from pandas import DataFrame, read_csv

from .. import WORKDIR

def parse_phenopacket(json_path: str) -> DataFrame:
    # Return a list of HPO terms in dataframe[id, name]
    from json import loads
    with open(json_path) as file:
        data = file.read()
    data_json = loads(data)
    hpo_data = []
    for entry in data_json['phenotypicFeatures']:
        hpo_entry = entry['type']
        hpo_data.append({'id': hpo_entry['id'], 'name': hpo_entry['label']})
    df = DataFrame.from_dict(hpo_data)
    return df

def parse_pheno_csv(csv_path: str) -> DataFrame:
    # Read CSV with colums id, name, tab separated
    # Return a list of HPO terms in dataframe[id, name]
    df: DataFrame = read_csv(csv_path, sep='\t')
    return df

def prepare_clinical_cases(cases_dir: str) -> dict:
    """
    Expected catalog structure:
    phenodata
    ├── acceptedmonkey
    │ ├── acceptedmonkey_gatkcomb_rhocall_norm_af_mt_frqf_cadd_vep_parsed_ranked.selected.vcf.gz
    │ ├── acceptedmonkey_gatkcomb_rhocall_norm_af_mt_frqf_cadd_vep_parsed_ranked.vcf.gz
    │ └── F0010719-2_2026-02-17_scout_phenopacket.json
    └── supergoblin
        ├── gene.txt
        ├── pheno.csv
        ├── superbgoblin_gatkcomb_rhocall_norm_af_mt_frqf_cadd_vep_parsed_ranked.selected.vcf.gz
        ├── superbgoblin_gatkcomb_rhocall_norm_af_mt_frqf_cadd_vep_parsed_ranked.vcf.gz

    :param cases_dir: Path to phenodata dir
    :return: Structurized dict per case
    """

    case_names = os.listdir(cases_dir)
    case_names = [case_name for case_name in case_names if not 'README.md' in case_name]

    # Structurize data into dict
    case_spec = dict()
    for case_name in case_names:
        case_dir = cases_dir + '/' + case_name
        vcf = glob(case_dir + '/*ranked.selected.vcf.gz')  # FIXME: Use all of data, not just selected variants
        pheno = glob(case_dir + '/*csv')
        phenopacket = glob(case_dir + '/*phenopacket.json')
        gene = glob(case_dir + '/gene.txt')
        case_meta = {
            'vcf': vcf[0],
            'pheno_csv': pheno[0] if len(pheno) > 0 else None,
            'phenopacket': phenopacket[0] if len(phenopacket) > 0 else None,
            'ground_truth_gene_txt': gene[0] if len(gene) > 0 else None
        }
        case_spec.update({case_name: case_meta})

    # Parse the clinical HPO terms into case_spec
    for case_name, case_meta in case_spec.items():
        if case_meta['phenopacket']:
            df_hpo_terms = parse_phenopacket(case_meta['phenopacket'])
        elif case_meta['pheno_csv']:
            df_hpo_terms = parse_pheno_csv(case_meta['pheno_csv'])
        else:
            raise ValueError('No phenotypic data')
        case_meta['clinically_relevant_hpo_terms'] = df_hpo_terms
        case_spec.update({case_name: case_meta})

    # Parse ground truth gene into case_spec
    for case_name, case_meta in case_spec.items():
        gene_name = None
        if case_meta['ground_truth_gene_txt']:
            with open(case_meta['ground_truth_gene_txt'], 'r') as fp:
                data = fp.read()
            assert len(data) > 0
            gene_name = data.replace('\n', '')
        case_meta['ground_truth_gene_name'] = gene_name
        case_spec.update({case_name: case_meta})

    return case_spec