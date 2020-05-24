import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.utils.tensorboard as tensorboard
import torchsummary
import tqdm
import xarray as xr

from dataset import split_datasets
from nn.metrics import StreamingMean


class NetMetrics:
    """This class is supposed to create a dataframe that collects, updates and saves to file the metrics of a model."""

    TRAIN_LOSS = 't. loss'
    VALIDATION_LOSS = 'v. loss'

    def __init__(self, t: tqdm.tqdm, metrics_path, tboard_dir):
        """
        """
        self.metrics_path = metrics_path
        self.df = pd.DataFrame(columns=[
            self.TRAIN_LOSS, self.VALIDATION_LOSS
        ])

        self.writer = tensorboard.SummaryWriter(tboard_dir)

        self.t = t

    def add_graph(self, model, model_device, train_loader):
        """

        :param model:
        :param model_device:
        :param train_loader:
        """
        # Extract the inputs of the first batch from the DataLoader, since PyTorch
        # needs an input tensor to correctly trace the model.
        sensors, goals, _ = next(iter(train_loader))

        # Ensure that the inputs reside on the same device as the model
        sensors = sensors.to(model_device)
        goals = goals.to(model_device)

        self.writer.add_graph(model, (sensors, goals))

    def update(self, epoch, train_loss, valid_loss, patience_lost):
        """

        :param epoch
        :param train_loss
        :param valid_loss
        """
        metrics = {self.TRAIN_LOSS: float(train_loss), self.VALIDATION_LOSS: float(valid_loss)}
        self.df = self.df.append(metrics, ignore_index=True)

        self.writer.add_scalar('loss/train', train_loss, epoch)
        self.writer.add_scalar('loss/validation', valid_loss, epoch)

        self.t.set_postfix(metrics, patience_lost=patience_lost)

    def finalize(self):
        self.df.to_pickle(self.metrics_path)
        self.writer.close()


def to_torch_loader(dataset, **kwargs):
    """

    :param dataset:
    :param kwargs:
    :return:
    """
    scanner_image = dataset.scanner_image
    scanner_distances = dataset.scanner_distances
    wheel_target_speeds = dataset.wheel_target_speeds
    goal_positions = dataset.goal_position
    goal_angles = dataset.goal_angle

    # Add a new 'channels' dimension to scanner_distances so it can be
    # concatenated with scanner_image
    scanner_distances = scanner_distances.expand_dims(channel=['d'], axis=-1)

    # Concatenate the two variables to a single array and transpose the dimensions
    # to match the PyTorch convention of samples ⨉ channels ⨉ angles
    scanner_data = xr.concat([scanner_image, scanner_distances], 'channel')
    scanner_data = scanner_data.transpose('sample', 'channel', 'scanner_angle')

    goal_angles = goal_angles.expand_dims(dict(axis=['theta']), axis=-1)
    goal_data = xr.concat([goal_positions, goal_angles], 'axis')

    # FIXME: maybe save directly as float32?
    sensors = torch.as_tensor(scanner_data.data, dtype=torch.float)
    goals = torch.as_tensor(goal_data.data, dtype=torch.float)
    targets = torch.as_tensor(wheel_target_speeds.data, dtype=torch.float)

    dataset = data.TensorDataset(sensors, goals, targets)
    loader = data.DataLoader(dataset, **kwargs)

    return loader


def create_network(arch, dropout):
    if arch == "convnet":
        from nn.convnet import ConvNet
        return ConvNet(dropout)
    elif arch == "convnet_maxpool":
        from nn.convnet import ConvNet_MaxPool
        return ConvNet_MaxPool(dropout)
    elif arch == "convnet_maxpool_goal":
        from nn.convnet import ConvNet_MaxPool_Goal
        return ConvNet_MaxPool_Goal(dropout)
    else:
        raise ValueError("Unknown network architecture '%s'" % arch)


def save_network(model_dir, net):
    """

    :param model_dir:
    :param net:
    """
    model_path = os.path.join(model_dir, 'model.pt')
    torch.save(net, model_path)


def load_network(model_dir, device='cpu'):
    """

    :param model_dir:
    :param device:
    :return net:
    """
    model_path = os.path.join(model_dir, 'model.pt')
    net = torch.load(model_path, map_location=device)

    # Convert old versions of the network to the current format
    net = migrate_network(net)

    # Ensure that the network is loaded in evaluation mode by default.
    net.eval()

    return net


def migrate_network(net):
    """
        Support loading old versions of the networks, by migrating them to the
        current version.

    :param net: a network in a potentially old format
    :return: network converted to the latest format
    """

    from nn.convnet import ConvNet, ConvNet_MaxPool

    if isinstance(net, ConvNet) or isinstance(net, ConvNet_MaxPool):
        if not hasattr(net, 'drop1'):
            setattr(net, 'drop1', nn.Identity())

        if not hasattr(net, 'drop2'):
            setattr(net, 'drop2', nn.Identity())

    return net


def train_net(dataset, splits, model_dir, metrics_path, tboard_dir,
              arch, n_epochs=500, lr=0.001, batch_size=2**14, loss='mse', dropout=0.0):
    """

    :param dataset:
    :param splits:
    :param model_dir:
    :param metrics_path:
    :param tboard_dir:
    :param arch:
    :param n_epochs:
    :param lr:
    :param batch_size:
    :param loss:
    :param dropout:
    :return:
    """
    train, validation, _ = split_datasets(dataset, splits)

    train_loader = to_torch_loader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = to_torch_loader(validation, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Create the neural network and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = create_network(arch, dropout)
    net.to(device)

    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'smooth_l1':
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError("Unsupported loss function '%s'." % loss)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Print model information
    print("Device:", device)
    print("Loss function:", loss)
    print("Dropout: %s" % ('off' if dropout == 0.0 else 'on (p=%g)' % dropout))
    # torchsummary.summary(net, input_size=[[4, 180], [3]])
    print()

    # Support objects for metrics and validation
    train_loss = StreamingMean()

    validator = NetValidator(valid_loader, criterion, device)

    t = tqdm.trange(n_epochs, unit='epoch')

    metrics = NetMetrics(t, metrics_path, tboard_dir)
    metrics.add_graph(net, device, train_loader)

    stopping = EarlyStopping()

    # Main training loop
    for epoch in t:
        # Re-enable training mode, which is disabled by the evaluation
        # Turns on dropout, batch normalization updates, …
        net.train()
        train_loss.reset()

        for batch in train_loader:
            sensors, goals, targets = (tensor.to(device) for tensor in batch)

            # Reset all the stored gradients
            optimizer.zero_grad()

            # Perform forward, backward and optimization steps
            outputs = net(sensors, goals)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Accumulate metrics across batches
            train_loss.update(loss, sensors.shape[0])

        # Perform model validation
        valid_loss = validator.validate(net)

        # Check early stopping
        should_stop, patience_lost = stopping.should_stop(net, valid_loss, epoch)

        # Record the metrics for the current epoch
        metrics.update(epoch, train_loss.mean, valid_loss, patience_lost)

        if should_stop:
            print("Interrupting training early. Validation loss hasn't improved in %d epochs." % stopping.patience)
            break

    # Restore the model with the best validation loss
    best_epoch = stopping.restore_best_net(net)
    print("Restoring best network from epoch %d." % best_epoch)

    # Save the final model to file.
    save_network(model_dir, net)
    metrics.finalize()


class NetValidator:
    def __init__(self, valid_loader: data.DataLoader, criterion, device):
        """

        :param valid_loader:
        :param criterion:
        :param device:
        """
        self.criterion = criterion
        self.valid_loader = valid_loader
        self.device = device

        self.valid_loss = StreamingMean()

    def validate(self, net):
        """

        :param net:
        :return:
        """
        with torch.no_grad():
            # Switch to evaluation mode: disable dropout, batch normalization updates, …
            net.eval()
            self.valid_loss.reset()

            for batch in self.valid_loader:
                sensors, goals, targets = (tensor.to(self.device) for tensor in batch)

                outputs = net(sensors, goals)
                loss = self.criterion(outputs, targets)

                self.valid_loss.update(loss, sensors.shape[0])

        return self.valid_loss.mean


class EarlyStopping:
    """
    Implement early stopping by keeping track of the model with the best validation
    loss seen so far. If validation loss doesn't improve for `patience` epochs in a
    row, interrupt the training.
    """
    def __init__(self, patience=20, rel_tolerance=1.05):
        """

        :param patience:
        """
        self.patience = patience
        self.rel_tolerance = rel_tolerance

        self.best_loss = np.inf
        self.best_epoch = None
        self.best_net = None

        self.patience_lost = 0

    def should_stop(self, net, loss, epoch):
        """

        :param net:
        :param loss:
        :param epoch:
        :return:
        """
        if loss <= self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.best_net = net.state_dict()

        if loss <= self.rel_tolerance * self.best_loss:
            self.patience_lost = 0
        else:
            self.patience_lost += 1

        should_stop = (self.patience_lost == self.patience)

        return should_stop, self.patience_lost

    def restore_best_net(self, net):
        """

        :param net:
        :return:
        """
        net.load_state_dict(self.best_net)

        return self.best_epoch
