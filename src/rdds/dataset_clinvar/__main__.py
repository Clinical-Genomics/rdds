import argparse

from .clinvar import Clinvar

parser = argparse.ArgumentParser(prog='CLINVAR database adaptor',
                                 description='Downloads CLINVAR data')

parser.add_argument('mode',
                    choices=['download-preprocess'],
                    help='Main operating mode')

args = parser.parse_args()
clinvar = Clinvar()

if args.mode == 'download-preprocess':
    # TODO: Clean data if already exist
    clinvar.download()
    clinvar.preprocess()
