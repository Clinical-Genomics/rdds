from enum import IntEnum

"""
SO terms listed in decreasing order of impact as of
https://grch37.ensembl.org/info/genome/variation/prediction/predicted_data.html#consequences
date 2024-10-26

The higher the INT value, the worse impact on transcript.
"""

SO_TERMS = IntEnum('SO_TERMS',
[
    # HIGH impact
    ('transcript_ablation', 41),
    ('splice_acceptor_variant', 40),
    ('splice_donor_variant', 39),
    ('stop_gained', 38),
    ('frameshift_variant', 37),
    ('stop_lost', 36),
    ('start_lost', 35),
    ('transcript_amplification', 34),
    ('feature_elongation', 33),
    ('feature_truncation', 32),
    # MODERATE impact
    ('inframe_insertion', 31),
    ('inframe_deletion', 30),
    ('missense_variant', 29),
    ('protein_altering_variant', 28),
    # LOW impact
    ('splice_donor_5th_base_variant', 27),
    ('splice_region_variant', 26),
    ('splice_donor_region_variant', 25),
    ('splice_polypyrimidine_tract_variant', 24),
    ('incomplete_terminal_codon_variant', 23),
    ('start_retained_variant', 22),
    ('stop_retained_variant', 21),
    ('synonymous_variant', 20),
    # MODIFIER impact
    ('coding_sequence_variant', 19),
    ('mature_miRNA_variant', 18),
    ('5_prime_UTR_variant', 17),
    ('3_prime_UTR_variant', 16),
    ('non_coding_transcript_exon_variant', 15),
    ('intron_variant', 14),
    ('NMD_transcript_variant', 13),
    ('non_coding_transcript_variant', 12),
    ('coding_transcript_variant', 11),
    ('upstream_gene_variant', 10),
    ('downstream_gene_variant', 9),
    ('TFBS_ablation', 8),
    ('TFBS_amplification', 7),
    ('TF_binding_site_variant', 6),
    ('regulatory_region_ablation', 5),
    ('regulatory_region_amplification', 4),
    ('regulatory_region_variant', 3),
    ('intergenic_variant', 2),
    ('sequence_variant', 1)
    ]
)