import argparse

from .clinvar import Clinvar

parser = argparse.ArgumentParser(prog='CLINVAR database adaptor',
                                 description='Downloads CLINVAR data')

parser.add_argument('mode',
                    choices=['data-download'],
                    help='Main operating mode')

args = parser.parse_args()

if args.mode == 'data-download':
    # TODO: Clean data if already exist
    clinvar: Clinvar = Clinvar()
    clinvar.download()
else:
    raise NotImplemented(f'Unknown mode {args.mode}')
