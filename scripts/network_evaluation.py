import torch
import numpy as np
import pandas as pd

from neural_networks import split_datasets, to_torch_loader, load_network
from plots import plot_losses, plot_target_distribution, plot_regressor, plot_wheel_speed_over_time, \
    plot_velocities_over_time, plot_losses_distribution


def get_predictions(net, valid_loader, device):
    """

    :param net:
    :param valid_loader:
    :param device:
    :return predictions, groundtruth:
    """
    predictions = []
    groundtruth = []
    losses = []

    # The unreduced losses are used to generate the plot of their distribution.
    criterion = torch.nn.MSELoss(reduction='none')

    with torch.no_grad():
        for batch in valid_loader:
            inputs, targets = (tensor.to(device) for tensor in batch)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            predictions.append(outputs)
            groundtruth.append(targets)
            losses.append(loss)

        # Concatenate the per-batch tensors in a single per-dataset tensor.
        predictions = torch.cat(predictions)
        groundtruth = torch.cat(groundtruth)
        losses = torch.cat(losses)

    return predictions, groundtruth, losses


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

    v_prediction, v_groundtruth, v_losses = get_predictions(net, valid_loader, device)
    t_prediction, t_groundtruth, t_losses = get_predictions(net, train_loader, device)

    predictions = [t_prediction, v_prediction]
    groundtruths = [t_groundtruth, v_groundtruth]
    unreduced_losses = [t_losses, v_losses]

    network_plots(img_dir_model, losses, predictions, groundtruths, unreduced_losses)


def network_plots(model_img, losses_per_epoch, predictions, groundtruths, unreduced_losses):
    """
    :param model_img
    :param losses_per_epoch:
    :param predictions
    :param groundtruths:
    :param unreduced_losses:
    """
    labels = ['train', 'validation']

    # Plot train and validation losses over training epochs
    plot_losses(*losses_per_epoch, model_img, 'loss')

    # Plot histogram with the distribution of losses
    unreduced_losses = [losses.numpy() for losses in unreduced_losses]
    plot_losses_distribution(*unreduced_losses, model_img, 'losses-distribution')

    for idx, el in enumerate(labels):
        # Plot histogram wheel target speeds
        y_g = groundtruths[idx].numpy()
        y_p = predictions[idx].numpy()

        plot_target_distribution(y_g, y_p, model_img, 'distribution-target-%s' % el)

        # Plot wheel target speeds over time
        plot_wheel_speed_over_time(y_g, y_p, model_img, 'wheel-speed-over-time-%s' % el)

        # Plot linear and angular velocity over time
        plot_velocities_over_time(y_g, y_p, model_img, 'velocities-over-time-%s' % el)

        # Evaluate prediction of the learned controller to the omniscient groundtruth
        # Plot R^2 of the regressor between prediction and ground truth on the validation set
        plot_regressor(y_g, y_p, model_img, 'regression-%s' % el)
