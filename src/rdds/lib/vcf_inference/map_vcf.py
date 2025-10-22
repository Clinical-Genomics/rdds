import os
from logging import Logger
from rdds.lib import slurm
from rdds.lib.process_pool import ProcessPool
from rdds.lib.list_dir import list_dir
from tempfile import mkdtemp
from subprocess import check_call
import progressbar
import math
from  typing import Callable
from rdds.lib.vcf import VCFReader


def map_vcf(vcf_file_path: str,
            fn: Callable,
            cpu_cores: int,
            workdir: str,
            logger: Logger,
            fn_kwargs: dict = {},
            max_batch_size_per_worker: int = int(10E4),
            ) -> str:
    """
    Apply fn to shards of VCF based on variant index. Batch sizes are dependent on CPU count and
    max_batch_size_per_worker.

    :param vcf_file_path: The path to the VCF file
    :param fn: A callable that must at least accept keywords:
        - vcf_file_path: str, the path to VCF to process
        - subprocess_work_dir: str, tmp dir for workers (shared across workers)
        - variant_index_start: int, starting index of variant to process in VCF
        - variant_index_stop: int, last index of variant to process in VCF
        The fn should write a .vcf to subprocess work dir with the following file name:
        [variant_index_start].vcf
    :param cpu_cores: Maximum amount of CPU cores to allocate to job
    :param max_batch_size_per_worker: Amount of variants per worker subprocess task (partition due to limited RAM)
    :return: Prints annotated file path on stdout and in return value
    """

    # TODO: cyvcf2 currently does not support writing string type format fields with number>1.

    if not '.vcf' in vcf_file_path:
        raise ValueError(f'Expected a VCF but got {vcf_file_path}')
    new_file_name = os.path.basename(vcf_file_path).replace('.vcf', '-predictions.vcf')  # FIXME
    annotated_vcf_file_path = os.path.join(os.path.dirname(vcf_file_path), new_file_name)
    if annotated_vcf_file_path == vcf_file_path:
        raise ValueError(f'Won\'t overwrite existing VCF file')

    if not os.path.exists(workdir):
        os.makedirs(workdir, exist_ok=True)
    subprocess_work_dir = mkdtemp(prefix='map-vcf-', dir=workdir)

    # Count number of variants in input VCF
    vcf_reader = VCFReader(vcf_file_path)
    n_variants = vcf_reader.number_of_variants
    vcf_reader.close()
    del vcf_reader

    n_workers = min(cpu_cores, slurm.cpu_count())
    batch_size = int(math.ceil((n_variants / float(n_workers))))
    batch_size = max_batch_size_per_worker if batch_size > max_batch_size_per_worker else batch_size
    subprocess_kwargs = []
    worker_names = []
    for variant_idx in range(0, n_variants, batch_size):
        worker_kwargs = {
            'subprocess_work_dir': subprocess_work_dir,
            'vcf_file_path': vcf_file_path,
            'variant_index_start': variant_idx,
            'variant_index_stop': variant_idx + batch_size
        }
        worker_kwargs.update(fn_kwargs)
        subprocess_kwargs.append(worker_kwargs)
        worker_names.append(f"{variant_idx}-{variant_idx+batch_size}")
    logger.info(f'Worker load: {n_workers} workers \
with {batch_size:.0E} variants per worker, \
totalling {len(subprocess_kwargs)} worker tasks')
    pool = ProcessPool(function=fn,
                       kwargs=subprocess_kwargs,
                       process_names=worker_names,
                       workers=n_workers)
    task_queue = pool.run_async()
    pbar = progressbar.ProgressBar(max_value=len(subprocess_kwargs),
                                   prefix='Total processing progress ')
    completed_tasks = []
    while True:
        task = task_queue.get(timeout=60*60*5)  # Raises Empty exception if no data after timeout
        if task.process.exitcode != 0:
            raise ValueError(f'Task failed: {task}')
            break
        completed_tasks.append(task)
        if len(completed_tasks) == len(subprocess_kwargs):
            break
        pbar.update(len(completed_tasks))
    pbar.finish()
    pool.close()

    # Merge VCFs to produce final output file
    # TODO: Check that all files are non-empty and submitted jobs
    vcf_parts = list(list_dir(directory_path=subprocess_work_dir))
    vcf_parts = sorted(vcf_parts, key=lambda path: int(os.path.basename(path.rstrip('.vcf'))))
    cmdline = "bcftools concat --no-version "
    for vcf_part in vcf_parts:
        cmdline += "%s " % vcf_part
    cmdline += " > %s" % annotated_vcf_file_path
    check_call(cmdline, shell=True)
    [os.remove(vcf_part) for vcf_part in vcf_parts]
    os.rmdir(subprocess_work_dir)
    logger.info(f'Completed. Output file: {annotated_vcf_file_path}')
    return annotated_vcf_file_path
