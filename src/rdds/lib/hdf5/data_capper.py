from h5py import File as Hd5File
import os

def cap_hd5_file(file_path: str):
    if not '.hd5' in file_path:
        raise ValueError(f'Expected a HDF5 file')
    output_file_path = file_path.replace('.hd5','-cap.hd5')
    if output_file_path == file_path:
        raise ValueError(f'Won\'t overwrite existing file')

    input_file = Hd5File(file_path, 'r')
    output_file = Hd5File(output_file_path, 'w')
    for group_name in input_file.keys():
        new_group = output_file.create_group(group_name)
        for dataset in group.keys():

