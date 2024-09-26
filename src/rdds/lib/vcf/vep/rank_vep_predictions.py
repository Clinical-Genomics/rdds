from typing import List

from .variant_effect_predictor_entry import VariantEffectPredictorEntry


def rank_vep_predictions(vep_csq_keys: List[str],
                         csq_data: str) -> VariantEffectPredictorEntry:
    """
    Reduce VEP variant effect predictions to the most harmful in a set.

    The VEP INFO/CSQ field is made up of several entries, relating to
    effect(s) this variant might have.

    This methods selects the most harmful effect as predicted by VEP
    and returns this sub entry to the caller.
    """
    if not isinstance(csq_data, str):
        raise ValueError(f'Expected str type got {type(csq_data)}')

    # VEP predicted effects separated by ',' character
    vep_predictions: List[str] = csq_data.split(',')

    if len(vep_predictions) == 0:
        raise ValueError(f'Expected at least one transcript in INFO/CSQ field, got none')

    vep_entries: List[VariantEffectPredictorEntry] = []

    for vep_prediction in vep_predictions:

        # Subdata split on | character per computed effect basis
        csq_data_subfields: List[str] = vep_prediction.split('|')

        if not len(csq_data_subfields) == len(vep_csq_keys):
            raise ValueError(f'Transcript information number of entries {csq_data_subfields}\
does not match key names {vep_csq_keys}')

        rankable_transcript = VariantEffectPredictorEntry()
        for key, value in zip(vep_csq_keys, csq_data_subfields):
            rankable_transcript.__setattr__(key, value)

        vep_entries.append(rankable_transcript)

    # Rank effects based on predicted significance, the higher the amount the more significant
    vep_entries.sort(key=lambda rankable_transcript: rankable_transcript.significance,
                     reverse=True)

    return vep_entries[0]
