import torch
import numpy as np
import pandas as pd

from neural_networks import split_datasets, to_torch_loader, load_network
from plots import plot_losses, plot_target_distribution, plot_regressor, plot_wheel_speed_over_time, \
    plot_velocities_over_time


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
    t_loss = losses.loc[:, 't. loss']
    v_loss = losses.loc[:, 'v. loss']
    losses = [t_loss, v_loss]

    train, validation, _ = split_datasets(dataset, splits)
    valid_loader = to_torch_loader(validation, batch_size=1024, shuffle=False, pin_memory=True)
    train_loader = to_torch_loader(train, batch_size=1024, shuffle=False, pin_memory=True)

    # Forcing evaluation on the CPU because part of the code below don't expect
    # GPU tensors. I think evaluating on the CPU makes sense anyway, since speed
    # is very fast already and this leaves the GPU free to concurrently train
    # another model. TODO: re-evaluate
    device = torch.device("cpu")
    net = load_network(model_dir, device)

    v_prediction, v_groundtruth = get_predictions(net, valid_loader, device)
    t_prediction, t_groundtruth = get_predictions(net, train_loader, device)

    predictions = [t_prediction, v_prediction]
    groundtruths = [t_groundtruth, v_groundtruth]
    network_plots(img_dir_model, losses, predictions, groundtruths)


def network_plots(model_img, losses, predictions, groundtruths):
    """
    :param model_img
    :param losses:
    :param predictions
    :param groundtruths:
    """
    labels = ['train', 'validation']

    # Plot train and validation losses
    plot_losses(losses[0], losses[1], model_img, 'loss')

    for idx, el in enumerate(labels):
        # Plot histogram wheel target speeds
        y_g = np.concatenate([gt.numpy() for gt in groundtruths[idx]])
        y_p = np.concatenate([p.detach().numpy() for p in predictions[idx]])

        plot_target_distribution(y_g, y_p, model_img, 'distribution-target-%s' % el)

        # Plot wheel target speeds over time
        plot_wheel_speed_over_time(y_g, y_p, model_img, 'wheel-speed-over-time-%s' % el)

        # Plot linear and angular velocity over time
        plot_velocities_over_time(y_g, y_p, model_img, 'velocities-over-time-%s' % el)

        # Evaluate prediction of the learned controller to the omniscient groundtruth
        # Plot R^2 of the regressor between prediction and ground truth on the validation set
        plot_regressor(y_g, y_p, model_img, 'regression-%s' % el)
