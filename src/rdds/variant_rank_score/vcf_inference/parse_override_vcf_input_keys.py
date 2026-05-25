from typing import Dict


def parse_override_vcf_input_keys(keys) -> Dict[str, str]:
    """
    Parse [OLD_KEY:NEW_KEY],[...],... syntax from cli to override model input annotations.
    :param str: The complete string
    :return: A dict where the key is the old vcf key, and value is the new vcf key
    """
    if keys is None:
        return None
    assert isinstance(keys, str), f'Expected keys to be a string but got {type(keys)} {keys}'
    assert len(keys) > 0, 'Expected keys to not be empty string'
    override_vcf_input_keys: dict[str, str] = dict()
    key_value_pairs = keys.split(',')
    for key_value_pair in key_value_pairs:
        key, value = key_value_pair.split(':')
        override_vcf_input_keys.update({key: value})
    return override_vcf_input_keys
