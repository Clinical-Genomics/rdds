def is_below_normalisation_low_bound(rank_score: float,
                                     rank_score_normalized: float,
                                     rank_score_normalization_low_bound: float) -> bool:
    """
    RankScore(s) below rank_score_normalized lower bound are capped to the lower bound of the normalized rank score,
    according to patch in genmod:
        commit 060942c12d33ad6b1501d2c3e6652bb75b4f00ad (origin/rankscore-normalization)
            Author: Tor Björgen <tor.bjorgen@scilifelab.se>
            Date:   Tue Jun 20 09:20:28 2023 +0200

            Compound scoring: Cap rankscore values to (min, ) range

            * compound scoring: When correcting rankscore, cap to min bound
            * Update READMEs on compound scoring

    Expect a discrepancy between rank score reference and new rank score from genmod with above patch.
    :param rank_score: The rank score to be tested, to see if it's below (LOW, HIGH) normalisation bound
    :param rank_score_normalized: Rank Score Normalised value
    :param rank_score_normalisation_low_bound: Low bound of rank score normalisation
    :return: True if RankScore is below RankScoreNormalisation lower bound
    """
    if rank_score < rank_score_normalization_low_bound and rank_score_normalized == rank_score_normalization_low_bound:
        return True
    return False