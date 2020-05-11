import torch
import numpy as np
import pandas as pd

from neural_networks import split_datasets, to_torch_loader, load_network
from plots import plot_losses, plot_target_distribution, plot_regressor


def get_predictions(net, valid_loader, device):
    """

    :param net:
    :param valid_loader:
    :param device:
    :return prediction, groundtruth:
    """
    prediction = []
    groundtruth = []

    for batch in valid_loader:
        inputs, targets = (tensor.to(device) for tensor in batch)
        prediction.append(net(inputs))
        groundtruth.append(targets)

    return prediction, groundtruth


def evaluate_net(dataset, splits, model_dir, img_dir_model, file_metrics):
    """

    :param dataset:
    :param splits:
    :param model_dir:
    :param model:
    :param img_dir_model:
    :param file_metrics:
    :return:
    """
    losses = pd.read_pickle(file_metrics)
    training_loss = losses.loc[:, 't. loss']
    validation_loss = losses.loc[:, 'v. loss']

    train, validation, _ = split_datasets(dataset, splits)
    valid_loader = to_torch_loader(validation, batch_size=1024, shuffle=False, pin_memory=True)

    # Forcing evaluation on the CPU because part of the code below don't expect
    # GPU tensors. I think evaluating on the CPU makes sense anyway, since speed
    # is very fast already and this leaves the GPU free to concurrently train
    # another model. TODO: re-evaluate
    device = torch.device("cpu")
    net = load_network(model_dir, device)

    prediction, groundtruth = get_predictions(net, valid_loader, device)

    network_plots(img_dir_model, training_loss, validation_loss, groundtruth, prediction)


def network_plots(model_img, training_loss, validation_loss, groundtruth, prediction):
    """
    :param model_img
    :param model
    :param net_input
    :param prediction:
    :param training_loss:
    :param validation_loss:
    :param x_train:
    :param y_valid:
    :param groundtruth:
    """

    # Plot train and validation losses
    plot_losses(training_loss, validation_loss, model_img, 'loss')

    # Plot histogram wheel target speeds
    y_g = np.concatenate([gt.numpy() for gt in groundtruth])
    y_p = np.concatenate([p.detach().numpy() for p in prediction])  # FIXME p.detach()

    plot_target_distribution(y_g, y_p, model_img, 'distribution-target')

    # Evaluate prediction of the learned controller to the omniscient groundtruth
    # Plot R^2 of the regressor between prediction and ground truth on the validation set
    plot_regressor(y_g, y_p, model_img, 'regression')
