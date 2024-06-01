#!/usr/bin/env python3

"""
Script for visualizing MRI images using different explanatory methods.

Author: Angel Sevilla Molina
source: https://github.com/angelsevillamol/xai-parkinsons-cnn/blob/main/visualize_heatmaps.py
"""

import argparse
import os
import sys
from typing import Tuple
import torch
import nibabel as nib
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from config_reader import ConfigReader


N_FOLDS = 5
N_SPLITS = 30
MODELS = ['beta1', 'beta2', 'beta3', 'gamma', 'nominal']


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
      Parsed arguments from the CLI.
    """
    parser = argparse.ArgumentParser(
        description='Visualization of MRI images applying different '
                    'explanatory methods.'
        )
    parser.add_argument('--model', type=str, choices=MODELS, required=True,
                        help='Network model to visualize.')
    parser.add_argument('--fold', type=int, choices=range(N_FOLDS),
                        required=True, help='Fold number.')
    parser.add_argument('--split', type=int, choices=range(N_SPLITS),
                        required=True, help='Split number.')

    return parser.parse_args()


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalizes a tensor to a range between 0 and 1.

    Args:
      tensor: the tensor to normalize.

    Returns:
      The normalized tensor.
    """
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    range_val = max_val - min_val
    if range_val < 1e-10:
        return torch.zeros_like(tensor)
    return (tensor - min_val) / range_val


def plot_image(
    ax: plt.Axes,
    image_t: torch.Tensor,
    cmap: str,
    overlay_t: torch.Tensor = None
) -> None:
    """Draws an image on a specific plot axis.

    Args:
      ax: the axis on which the image will be drawn.
      image_t: the tensor containing the image to be visualized.
      cmap: the color map to visualize the image.
      overlay_t: the tensor containing data to overlay on the main image.
    """
    # Convert the tensor to numpy for visualization
    data = image_t.cpu().numpy()

    # If there is no overlay tensor, show the main image
    if overlay_t is None:
        ax.imshow(data.T, cmap=cmap, vmin=0, vmax=1, origin='lower')
    # If there is an overlay tensor, show the image and overlay the heatmap
    else:
        overlay_data = overlay_t.cpu().numpy()
        ax.imshow(data.T, cmap='gray', origin='lower')
        ax.imshow(overlay_data.T, cmap=cmap,
                  alpha=normalize_tensor(overlay_t).T,
                  vmin=0, vmax=1, origin='lower')
    ax.axis('off')


def extract_planes(
    image_t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extracts axial, sagittal, and coronal planes from an MRI image
    tensor.

    Args:
      image_t: the tensor containing the 3D MRI image.

    Returns:
      A set of the three anatomical planes extracted from the tensor.
    """
    # Determine the midpoint for each dimension
    x_mid = image_t.shape[0] // 2
    y_mid = image_t.shape[1] // 2
    z_mid = image_t.shape[2] // 2
    return image_t[:, :, z_mid], image_t[x_mid, :, :], image_t[:, y_mid, :]


def visualize_mri(
    orig_t: torch.Tensor,
    gradcam_t: torch.Tensor,
    gbp_t: torch.Tensor,
    lrp_t: torch.Tensor,
    out_path: str
) -> None:
    """Generates comparative visualizations of MRI anatomical planes
    using different explanatory methods.

    Args:
      orig_t: the tensor of the original MRI image.
      gradcam_t: the tensor of the image obtained using Grad-CAM.
      gbp_t: the tensor of the image obtained using GBP.
      lrp_t: the tensor of the image obtained using LRP.
      out_path: Path where the visualized image will be saved.
    """
    # Define the parameters of the explanatory methods
    methods = [
        ('Original', 'gray', extract_planes(orig_t), [None, None, None]),
        ('Grad-CAM', 'jet', extract_planes(orig_t), extract_planes(gradcam_t)),
        ('GBP', 'jet', extract_planes(orig_t), extract_planes(gbp_t)),
        ('LRP', 'jet', extract_planes(orig_t), extract_planes(lrp_t))
    ]

    # Draw each plane in its respective axis
    fig, axes = plt.subplots(3, 4, figsize=(17, 12), constrained_layout=True)
    for col, (name, cmap, img_planes, overlay_planes) in enumerate(methods):
        for row, (img, overlay) in enumerate(zip(img_planes, overlay_planes)):
            plot_image(axes[row, col], img, cmap, overlay)
            if row == 0:
                axes[row, col].set_title(name)

    # Configure the color bar
    sm = ScalarMappable(cmap='jet', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[:, 3], aspect=20)

    # Save the plot as png
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()


def main() -> None:
    """Generates a visualization of MRI images applying different
    explanatory methods."""

    # Read arguments from the parser
    args = parse_arguments()

    # Read the paths from the configuration file
    try:
        config_reader = ConfigReader('config.ini')
        data_root_path = config_reader.get_data_root_path()
        heatmaps_path = config_reader.get_heatmaps_path()
        planes_path = config_reader.get_planes_path()
    except FileNotFoundError as error:
        sys.exit(f'Error: {error}')
    except KeyError as error:
        sys.exit(f'Error: {error}')

    # Load the directories with the heatmaps of each method
    split_dir = f'{args.model}/fold{args.fold}/split{args.split}'
    paths = {
        'gradcam': os.path.join(heatmaps_path, f'gradcam/{split_dir}'),
        'gbp': os.path.join(heatmaps_path, f'gbp/{split_dir}'),
        'lrp': os.path.join(heatmaps_path, f'lrp/{split_dir}'),
        'output': os.path.join(planes_path, f'{split_dir}')
    }
    os.makedirs(paths['output'], exist_ok=True)

    # For each heatmap
    for filename in os.listdir(paths['gradcam']):
        if filename.endswith('.pt'):
            file_paths = {method: os.path.join(path, filename)
                          for method, path in paths.items() if method != 'output'}
            # Check if the necessary files exist
            if all(os.path.exists(path) for path in file_paths.values()):
                # Load the NifTI image
                orig_filename = filename.replace('.pt', '.nii.gz')
                orig_path = os.path.join(data_root_path, orig_filename)
                orig_img = nib.load(orig_path)

                # Load the heatmap images
                orig_t = torch.tensor(orig_img.get_fdata())
                gradcam_t = torch.load(file_paths['gradcam'])
                gbp_t = torch.load(file_paths['gbp'])
                lrp_t = torch.load(file_paths['lrp'])

                # Generate a comparative image of the sections
                out_filename = filename.replace('.pt', '.png')
                out_path = os.path.join(paths['output'], out_filename)
                print(f'Generating image {out_path}')
                visualize_mri(orig_t, gradcam_t, gbp_t, lrp_t, out_path)
            # If any file is missing
            else:
                missing_files = [method
                                 for method, path in file_paths.items()
                                 if not os.path.exists(path)]
                sys.exit(f'Error: Missing files for {filename}: '
                     f'{", ".join(missing_files)}')


if __name__ == '__main__':
    main()
