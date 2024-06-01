#!/usr/bin/env python3

"""
Script for generating heatmaps using XAI techniques for a set of
previously trained convolutional neural networks.

Author: Angel Sevilla Molina
source: 
"""

import os
import random
from inspect import signature
from typing import Tuple, List
import posix_ipc
import numpy as np
import pandas as pd
import signac
import torch
import torch.utils.data as tdata
from condor import CondorEnvironment
from flow import FlowProject, directives

from config_reader import ConfigReader
from network import OrdinalNet, NominalNet, WrapperOrdinalNet
from xai_method import XAIMethod


# Read the paths from the configuration file
config_reader = ConfigReader('config.ini')
labels_path = config_reader.get_labels_path()
heatmaps_path = config_reader.get_heatmaps_path()

METHODS = ['gbp', 'gradcam', 'lrp']
project = signac.get_project()
project_data_semaphore_name = f'/{project.id}_project-data'


class AYRNACondorEnvironment(CondorEnvironment):
    """
    Environment for executing AYRNA projects on Condor.
    """
    hostname_pattern = r'srvrryc(arn|inf)\d{2}\.priv\.uco\.es$'
    template = 'custom-condor-submit.sh'


def determine_classifier_type(job : signac.contrib.job.Job) -> str:
    """Determines the type of classifier associated with the job.

    Args:
      job: the signac job.

    Returns:
      The classifier type. It can be 'beta1', 'beta2', 'beta3',
      'gamma', or 'nominal'.

    Raises:
      NotImplementedError: if the classifier type is not implemented.
    """
    xqu_to_beta = {
        0.65: '1',  # P(X < 0.5) = 0.75 and P(X < 0.65) = 0.9 (beta1)
        0.75: '2',  # P(X < 0.5) = 0.75 and P(X < 0.75) = 0.9 (beta2)
        0.85: '3'   # P(X < 0.5) = 0.75 and P(X < 0.85) = 0.9 (beta3)
    }
    if job.sp.classifier_type == 'ordinal':
        if 'ordinal_augment_beta_params' in job.sp:
            bp = job.sp.ordinal_augment_beta_params
            beta = xqu_to_beta.get(bp.xqu)
            classifier_type = f'beta{beta}'
        elif 'ordinal_augment_gamma_params' in job.sp:
            classifier_type = 'gamma'
        else:
            classifier_type = 'null'
            raise NotImplementedError('classifier type not implemented')
    elif job.sp.classifier_type == 'nominal':
        classifier_type = 'nominal'
    else:
        classifier_type = 'null'
        raise NotImplementedError('classifier type not implemented')
    return classifier_type


def get_split_names(sp : signac.core.attrdict.SyncedAttrDict) -> List[str]:
    """Gets the names of the splits contained in the job.

    Args:
      sp: the signac job statepoint.

    Returns:
      A list of split names.
    """
    with posix_ipc.Semaphore(project_data_semaphore_name,
                             flags=posix_ipc.O_CREAT,
                             initial_value=1):
        with project.data:
            split_names = [n for n in project.data[
                f'seed{sp.seed}/fold{sp.fold}/{sp.phase}'].keys()
                if n.startswith('split')]
    return split_names


def load_model_components(
    image_shape : Tuple[int, ...],
    n_channels : List[int],
    kernel_size : int,
    stride : int,
    hidden_size : int,
    n_classes : int,
    dropout_rate : float,
    learning_rate : float,
    classifier_type: str,
    device : torch.device
) -> torch.nn.Module:
    """Creates a neural network model with the corresponding
    hyperparameters.

    Args:
      image_shape: the shape of the input image.
      n_channels: the number of input channels.
      kernel_size: the size of the kernel.
      stride: the stride of the convolution.
      hidden_size: the size of the hidden layer.
      n_classes: the number of output classes.
      dropout_rate: the dropout rate.
      learning_rate: the learning rate.
      classifier_type: if the classifier is ordinal or nominal.
      device: the computing device.

    Returns:
      A neural network model prepared for training. The model is either
      nominal or ordinal, depending on the value of classifier_type.

    Raises:
      NotImplementedError: if the classifier type is not implemented.
    """
    if classifier_type == 'nominal':
        net = NominalNet(image_shape, n_channels, kernel_size, stride,
                         hidden_size, n_classes, dropout_rate)
    elif classifier_type == 'ordinal':
        net = OrdinalNet(image_shape, n_channels, kernel_size, stride,
                         hidden_size, n_classes, dropout_rate)
    else:
        raise NotImplementedError('classifier type not implemented')
    net = net.to(device)
    return net


def seed_from_str(s : str) -> int:
    """Generates a pseudorandom seed from a string.

    Args:
      s: the input string.

    Returns:
      A pseudorandom seed.
    """
    return hash(s) % (2 ** 32)


def determinism(seed : int) -> None:
    """Configures pseudorandom seeds for deterministic behavior of the
    experiment.

    Args:
      seed: the seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def prepare_model(
    job : signac.contrib.job.Job,
    split : str,
    device : torch.device
) -> torch.nn.Module:
    """Prepares the network for evaluation, and loads the weight values
    if the model is trained.

    Args:
      job: the signac job.
      split: indicates the split name.
      device: the computing device.

    Returns:
      The prepared neural network model.
    """
    split_seed = seed_from_str(f'{job.sp.seed}_{split}')
    determinism(split_seed)
    net = load_model_components(
        **{p: v for p, v in job.sp.items()
           if p in signature(load_model_components).parameters},
        image_shape=job.doc.image_shape,
        n_classes=job.doc.n_classes,
        device=device
    )

    if job.isfile(f'trained_state_{split}.pt'):
        net.load_state_dict(torch.load(job.fn(f'trained_state_{split}.pt'),
                                       map_location=device))
    net = net.to(device)
    net.eval()
    return net


def load_data(
    sp : signac.core.attrdict.SyncedAttrDict,
    split : str,
    device : torch.device
) -> Tuple[tdata.Dataset, list]:
    """Loads the input image dataset for testing from the
    corresponding split.

    Args:
      sp: the signac job statepoint.
      split: indicates the split name.
      device: the computing device.

    Returns:
      The selected NifTI image dataset and a list of indices indicating
      which original data each image corresponds to.
    """
    with posix_ipc.Semaphore(project_data_semaphore_name,
                             flags=posix_ipc.O_CREAT,
                             initial_value=1):
        with project.data:
            data = project.data
            if sp.phase == 'validation':
                test_idx = data[f'seed{sp.seed}/fold{sp.fold}/'
                                f'validation/{split}/test'][:]
            elif sp.phase == 'evaluation':
                test_idx = data[f'seed{sp.seed}/fold{sp.fold}/'
                                f'evaluation/test'][:]
            else:
                raise RuntimeError

            samples = data['samples']
            targets = data['targets']
            class_mapping = dict(sp.class_mapping)
            test_idx = sorted(test_idx)
            targets = np.array([class_mapping[t] for t in targets])
            test_samples = samples[test_idx]
            test_targets = targets[test_idx]
            test_ds = tdata.TensorDataset(
                torch.tensor(test_samples, device=device),
                torch.tensor(test_targets, device=device)
            )

    return test_ds, test_idx


def prepare_data(
    test_ds : tdata.Dataset,
    test_idx: list
) -> Tuple[tdata.DataLoader, list]:
    """Prepares the data for the experiment and retrieves the actual
    classes for each data.

    Args:
      test_ds: the test dataset.
      test_idx: list of indices.

    Returns:
      A DataLoader of the selected NifTI image dataset and a list of
      indices indicating the filename of the image.
    """
    test_dl = tdata.DataLoader(test_ds, batch_size=1, shuffle=False)
    labels_df = pd.read_csv(labels_path)
    filenames = labels_df['filename'].tolist()
    if test_idx is not None:
        filenames = [filenames[i] for i in test_idx]
    return test_dl, filenames


@FlowProject.operation
@directives(ngpu=1)
def generate_heatmaps(job : signac.contrib.job.Job) -> None:
    """Generates heatmaps applying XAI techniques for the model and
    datasets corresponding to a signac job.

    Args:
      job: the signac job.
    """
    if job.sp.phase != 'evaluation':
        return

    model = determine_classifier_type(job)
    split_names = get_split_names(job.sp)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # For each split
    for split in split_names:
        net = prepare_model(job, split, device)
        test_ds, test_idx = load_data(job.sp, split, device)
        test_dl, filenames = prepare_data(test_ds, test_idx)
        # If the model is ordinal, creates a wrapper to interpret the
        # results from the last layer
        if model != 'nominal':
            net = WrapperOrdinalNet(net)
            layer = net.original_net.convnet.convs[-2]
        else:
            layer = net.convnet.convs[-2]

        # For each explanatory method
        for method in METHODS:
            xai_method = XAIMethod(net, method, layer)
            split_dir = f'{method}/{model}/fold{job.sp.fold}/{split}'
            method_path = os.path.join(heatmaps_path, split_dir)
            os.makedirs(method_path, exist_ok=True)

            # For each data of the evaluation set
            print(f'Generating heatmaps in {method_path}')
            for idx, (sample, target) in enumerate(test_dl):
                input_t = sample
                input_t.requires_grad = True
                heatmap = xai_method.apply(input_t, target.item(),
                                           postprocess=True)
                filename = filenames[idx].split('.')[0] + '.pt'
                torch.save(heatmap, os.path.join(method_path, filename))


def main() -> None:
    """Main function to execute the FlowProject."""
    FlowProject().main()


if __name__ == '__main__':
    main()
