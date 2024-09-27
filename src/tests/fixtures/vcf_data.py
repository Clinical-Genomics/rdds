import pytest
from typing import Tuple
from os.path import dirname


@pytest.fixture
def minimally_annotated_ranked_mip_vcf() -> str:
    """
    Example data file containing ranked variants (minimally annotated)
    """
    return dirname(__file__)+'/minimally_annotated_ranked_mip.vcf.gz'


@pytest.fixture
def fully_annotated_unranked_mip_vcf() -> Tuple[int, str]:
    """
    Example data file containing fully annotated variants (excluding rank scores)
    :returns number of variants and path to file as tuple
    """
    _VCF_LENGTH = 1000
    _VCF_HEADER_LENGTH = 110
    _VCF_NR_VARIANTS = _VCF_LENGTH - _VCF_HEADER_LENGTH
    return _VCF_NR_VARIANTS, dirname(__file__) + '/fully_annotated_unranked_mip.vcf.gz'
