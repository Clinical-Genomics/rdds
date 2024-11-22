#!/usr/bin/env python3
import subprocess as sp
import json
from typing import List

if __name__ == '__main__':
    """
    Removes _ALL_ failed github actions runs, including unit tests, failed releases etc.
    """
    ret = sp.run('gh run list --status failure --limit 1000 --json databaseId',
                 shell=True,
                 capture_output=True,
                 check=True)
    failed_job_ids: bytes = ret.stdout
    jobs: List[dict] = json.loads(failed_job_ids)
    if len(jobs) == 0:
        print('No failed jobs to remove')
        exit(0)
    for entry in jobs:
        for _, job_id in entry.items():
            print(f'Removing job {job_id}')
            sp.check_call(f'gh run delete {job_id}', shell=True)