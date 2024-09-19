import argparse

from .giab import Giab

parser = argparse.ArgumentParser(prog='GIAB database adaptor',
                                 description='Downloads GIAB/AshkenazimTrio data')

parser.add_argument('mode',
                    choices=['download-preprocess'],
                    help='Main operating mode')

args = parser.parse_args()

if args.mode == 'download-preprocess':
    # TODO: Clean data if already exist
    giab = Giab()
    giab.download()
    giab.preprocess()
