import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.tensorboard as tensorboard
import torchsummary
import tqdm
import xarray as xr

from dataset import split_datasets


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Using BatchNorm somewhat improperly as an input normalization step.
        # This layer applies the transformation y = (x - mu) / std using per-channel
        # mean and stdev over the train set. After that, z = alpha * y + beta,
        # where alpha and beta are learned parameters, which allow the model to
        # learn the best scaling for the data.
        self.in_norm = nn.BatchNorm1d(num_features=4)

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=5,
                               stride=2, padding=2, padding_mode='circular')
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5,
                               stride=2, padding=2, padding_mode='circular')
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5,
                               stride=1, padding=2, padding_mode='circular')

        self.fc1 = nn.Linear(45 * 32, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, input):
        """

        :param input: input tensor with shape samples ⨉ channels ⨉ angles
        :return: output tensor with shape samples ⨉ wheels
        """
        net = self.in_norm(input)

        net = F.relu(self.conv1(net))
        net = F.relu(self.conv2(net))
        net = F.relu(self.conv3(net))

        net = torch.flatten(net, start_dim=1)

        net = F.relu(self.fc1(net))
        net = F.relu(self.fc2(net))
        output = self.fc3(net)

        return output


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
        # Extract the inputs of the first batch from the DataLoader, since PyTorch
        # needs an input tensor to correctly trace the model.
        inputs, _ = next(iter(train_loader))

        # Ensure that the inputs reside on the same device as the model
        inputs = inputs.to(model_device)

        self.writer.add_graph(model, inputs)

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


class StreamingMean:
    """
    Compute the (possibly weighted) mean of a sequence of values in streaming fashion.

    This class stores the current mean and current sum of the weights and updates
    them when a new data point comes in.

    This should have better stability than summing all samples and dividing at the
    end, since here the partial mean is always kept at the same scale as the samples.
    """
    def __init__(self):
        self.reset()

    def update(self, sample, weight=1.0):
        self._weights += weight
        self._mean += weight / self._weights * (sample - self._mean)

    def reset(self):
        self._weights = 0.0
        self._mean = 0.0

    @property
    def mean(self):
        return self._mean


def to_torch_loader(dataset, **kwargs):
    scanner_image = dataset.scanner_image
    scanner_distances = dataset.scanner_distances
    wheel_target_speeds = dataset.wheel_target_speeds

    # Add a new 'channels' dimension to scanner_distances so it can be
    # concatenated with scanner_image
    scanner_distances = scanner_distances.expand_dims(channel=['d'], axis=-1)

    # Concatenate the two variables to a single array and transpose the dimensions
    # to match the PyTorch convention of samples ⨉ channels ⨉ angles
    scanner_data = xr.concat([scanner_image, scanner_distances], 'channel')
    scanner_data = scanner_data.transpose('sample', 'channel', 'scanner_angle')

    # FIXME: maybe save directly as float32?
    inputs = torch.as_tensor(scanner_data.data, dtype=torch.float)
    targets = torch.as_tensor(wheel_target_speeds.data, dtype=torch.float)

    dataset = data.TensorDataset(inputs, targets)
    loader = data.DataLoader(dataset, **kwargs)

    return loader


def save_network(model_dir, net):
    model_path = os.path.join(model_dir, 'model.pt')
    torch.save(net, model_path)


def load_network(model_dir, device='cpu'):
    model_path = os.path.join(model_dir, 'model.pt')
    net = torch.load(model_path, map_location=device)

    return net


def train_net(dataset, splits, model_dir, metrics_path, tboard_dir, n_epochs=100, lr=0.01, batch_size=2**14):
    """

    :param dataset:
    :param splits:
    :param model_dir:
    :param metrics_path:
    :param tboard_dir:
    :param n_epochs:
    :param lr:
    :param batch_size:
    :return:
    """
    train, validation, _ = split_datasets(dataset, splits)

    train_loader = to_torch_loader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = to_torch_loader(validation, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Create the neural network and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = ConvNet()
    net.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Print model information
    print("Device:", device)
    torchsummary.summary(net, input_size=(4, 180))
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
        train_loss.reset()

        for batch in train_loader:
            inputs, targets = (tensor.to(device) for tensor in batch)

            # Reset all the stored gradients
            optimizer.zero_grad()

            # Perform forward, backward and optimization steps
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Accumulate metrics across batches
            train_loss.update(loss, inputs.shape[0])

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

    save_network(model_dir, net)
    metrics.finalize()


class NetValidator:
    def __init__(self, valid_loader: data.DataLoader, criterion, device):
        self.criterion = criterion
        self.valid_loader = valid_loader
        self.device = device

        self.valid_loss = StreamingMean()

    def validate(self, net):
        with torch.no_grad():
            self.valid_loss.reset()

            for batch in self.valid_loader:
                inputs, targets = (tensor.to(self.device) for tensor in batch)

                outputs = net(inputs)
                loss = self.criterion(outputs, targets)

                self.valid_loss.update(loss, inputs.shape[0])

        return self.valid_loss.mean


class EarlyStopping:
    """
    Implement early stopping by keeping track of the model with the best validation
    loss seen so far. If validation loss doesn't improve for `patience` epochs in a
    row, interrupt the training.
    """
    def __init__(self, patience=20):
        self.patience = patience

        self.best_loss = np.inf
        self.best_epoch = None
        self.best_net = None

    def should_stop(self, net, loss, epoch):
        if loss <= self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.best_net = net.state_dict()

        patience_lost = epoch - self.best_epoch
        should_stop = (patience_lost == self.patience)

        return should_stop, patience_lost

    def restore_best_net(self, net):
        net.load_state_dict(self.best_net)

        return self.best_epoch
