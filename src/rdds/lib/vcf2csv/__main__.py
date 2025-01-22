import argparse
from glob import glob

from .vcf2csv import Vcf2Csv


parser = argparse.ArgumentParser(prog='Vcf2Csv')
subparsers = parser.add_subparsers()

subparser = subparsers.add_parser('convert', help='Convert VCF to CSV format')
subparser.add_argument('vcf_file_path', help='Path to VCF file')
def _parse_vcf(args):
    vcf2csv = Vcf2Csv()
    vcf2csv.convert_vcf_to_csv(vcf_path=args.vcf_file_path)
subparser.set_defaults(func=_parse_vcf)

subparser = subparsers.add_parser('convert-multiple', help='Convert VCF to CSV format')
subparser.add_argument('vcf_glob_path', help='Globbed path to VCF files')
def _parse_vcf(args):
    vcf_paths = glob(args.vcf_glob_path)
    print(f'Converting {vcf_paths}')
    for vcf_path in vcf_paths:
        vcf2csv = Vcf2Csv()
        vcf2csv.convert_vcf_to_csv(vcf_path=vcf_path)
subparser.set_defaults(func=_parse_vcf)

args = parser.parse_args()
args.func(args)  # Callback to trigger func with args