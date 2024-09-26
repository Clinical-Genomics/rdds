from typing import Any

from .so_terms import SO_TERMS


class VariantEffectPredictorEntry:

    """

    Provides method to sort VEP effect predictions based on some estimated consequence.

    # TODO: Rank on Feature_type[Transcript, RegulatoryFeature, MotifFeature] as well?
    """

    @property
    def significance(self) -> int:
        """
        Return SUM(SO_TERMS) consequence ranking as the significance
        magnitude for this predicted effect.
        """
        try:
            consequences: str = self.Consequence  # VEP CSQ Consequence annotation [SO_TERM[&SO_TERM, ...]]
        except AttributeError:
            raise ValueError(f'Consequence not determined for this transcript!')
        significance = 0
        for consequence in consequences.split('&'):
            significance += SO_TERMS[consequence].value
        return significance

    @significance.setter
    def significance(self, value: None):
        raise ValueError('significance is immutable!')

    def __str__(self):
        """
        Printout helper function
        """
        s = None
        for attribute in dir(self):
            if '__' in attribute:
                continue
            value: Any = self.__getattribute__(attribute)
            s = f'{attribute}={value}' if s is None else s + f', {attribute}={value}'
        return s