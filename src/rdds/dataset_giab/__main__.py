import argparse

from .giab import Giab

parser = argparse.ArgumentParser(prog='GIAB database adaptor',
                                 description='Downloads GIAB/AshkenazimTrio data')

parser.add_argument('mode',
                    choices=['data-download'],
                    help='Main operating mode')

args = parser.parse_args()

if args.mode == 'data-download':
    # TODO: Clean data if already exist
    giab = Giab()
    giab.download()
else:
    raise NotImplemented(f'Unknown mode {args.mode}')
