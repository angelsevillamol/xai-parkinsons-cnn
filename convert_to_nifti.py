#!/usr/bin/env python3

"""
Converts a tensor saved in a file to an image in NifTI format.

Author: Angel Sevilla Molina
source: https://github.com/angelsevillamol/xai-parkinsons-cnn/blob/main/convert_to_nifti.py
"""

import argparse
import sys
import pickle
import torch
import numpy as np
import nibabel as nib


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
      Parsed arguments from the CLI.
    """
    parser = argparse.ArgumentParser(
        description='Converts a tensor saved in a file to an image in NifTI '
                    'format.'
        )
    parser.add_argument('input_path', type=str,
        help='Path to the input .pt file')
    parser.add_argument('output_path', type=str,
        help='Path to the output .nii.gz file')

    return parser.parse_args()


def load_nifti_metadata() -> dict:
    """Loads NIFTI metadata from a pickle file.

    Returns:
      A dictionary containing NIFTI metadata.

    Raises:
      FileNotFound: if the file with the metadata cannot be opened.
    """
    try:
        with open('nifti_metadata.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError('metadata not found')


def save_nifti(image_t : torch.Tensor, metadata : dict, output_path : str):
    """Saves a tensor as a NifTI image using loaded metadata.

    Args:
      image_t: the original image tensor.
      metadata: a dictionary containing NIFTI metadata.
      output_path: indicates the path where the image will be saved.
    """
    image_np = image_t.detach().cpu().numpy().astype(np.float32)
    img = nib.Nifti1Image(image_np, metadata['affine'], metadata['header'])
    nib.save(img, output_path)


def main():
    """Converts a tensor saved in a file to an image in NifTI format.
    """
    args = parse_arguments()
    try:
        image_t = torch.load(args.input_path)
        metadata = load_nifti_metadata()
        save_nifti(image_t, metadata, args.output_path)
    except Exception as error:
        sys.exit(f'Error: {error}')

if __name__ == '__main__':
    main()
