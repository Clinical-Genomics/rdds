import os


def cpu_count() -> int:
    """
    Get number of available cores in a SLURM environment.
    In SLURM, os.cpu_count returns node total count, not task allocated cores

    If SLURM_CPUS_PER_TASK is not set, return amount of cores on this host.

    :return: Amount of cores available
    """
    return int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
