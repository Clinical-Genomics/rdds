import argparse

from .mouse_human import MouseHuman

parser = argparse.ArgumentParser(prog='Mouse to Human dataset adaptor',
                                 description='Downloads Mouse to Human dataset')

parser.add_argument('mode',
                    choices=['download'],
                    help='Main operating mode')

args = parser.parse_args()
mouse_human = MouseHuman()

if args.mode == 'download':
    # TODO: Clean data if already exist
    mouse_human.download()
