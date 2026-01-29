import argparse

from .hgnc import HGNC

parser = argparse.ArgumentParser(prog='HGNC database adaptor',
                                 description='Downloads HGNC data')

parser.add_argument('mode',
                    choices=['download'],
                    help='Main operating mode')

args = parser.parse_args()
hgnc = HGNC()

if args.mode == 'download':
    # TODO: Clean data if already exist
    hgnc.download()
