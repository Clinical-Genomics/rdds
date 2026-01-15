import argparse

from .hpo import HPO

parser = argparse.ArgumentParser(prog='HPO database adaptor',
                                 description='Downloads HPO data')

parser.add_argument('mode',
                    choices=['download'],
                    help='Main operating mode')

args = parser.parse_args()
hpo = HPO()

if args.mode == 'download':
    # TODO: Clean data if already exist
    hpo.download()
