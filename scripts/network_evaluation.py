import torch
import numpy as np
import pandas as pd
from neural_networks import split_datasets, to_torch_loader
from plots import plot_losses


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


def evaluate_net(dataset, splits, model_dir, model, img_dir_model, file_metrics):
    net = torch.load('%s%s' % (model_dir, model), map_location=torch.device('cpu'))

    losses = pd.read_pickle(file_metrics)
    training_loss = losses.loc[:, 't. loss']
    validation_loss = losses.loc[:, 'v. loss']

    train, validation, _ = split_datasets(dataset, splits)
    valid_loader = to_torch_loader(validation, batch_size=1024, shuffle=False, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prediction, groundtruth = get_predictions(net, valid_loader, device)

    network_plots(img_dir_model, model, training_loss, validation_loss)


def network_plots(model_img, model, training_loss, validation_loss):
    """
    :param model_img
    :param dataset:
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
    title = 'Loss %s' % model
    file_name = 'loss-%s' % model
    plot_losses(training_loss, validation_loss, model_img, title, file_name)
