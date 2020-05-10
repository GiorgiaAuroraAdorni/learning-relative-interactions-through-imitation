import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.tensorboard as tensorboard
import tqdm
import xarray as xr

from dataset import split_datasets


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3,
                               stride=2, padding=1, padding_mode='circular')
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,
                               stride=2, padding=1, padding_mode='circular')
        self.maxp1 = nn.MaxPool1d(3)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3,
                               stride=1, padding=1, padding_mode='circular')

        self.fc1 = nn.Linear(15 * 32, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, input):
        """

        :param input: input tensor with shape samples ⨉ channels ⨉ angles
        :return: output tensor with shape samples ⨉ wheels
        """
        net = F.relu(self.conv1(input))
        net = F.relu(self.conv2(net))
        net = self.maxp1(net)
        net = F.relu(self.conv3(net))

        net = torch.flatten(net, start_dim=1)

        net = F.relu(self.fc1(net))
        output = self.fc2(net)

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

    def update(self, epoch, train_loss, valid_loss):
        """

        :param train_loss
        :param valid_loss
        """
        metrics = {self.TRAIN_LOSS: train_loss, self.VALIDATION_LOSS: valid_loss}
        self.df = self.df.append(metrics, ignore_index=True)

        self.writer.add_scalar('loss/train', train_loss, epoch)
        self.writer.add_scalar('loss/validation', valid_loss, epoch)

        self.t.set_postfix(metrics)

    def finalize(self):
        """

        :param out_file
        """
        self.df.to_pickle(self.metrics_path)
        self.writer.close()


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


def train_net(dataset, splits, model_dir, metrics_path, tboard_dir, n_epochs=100, lr=0.01, batch_size=1024):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = ConvNet()
    net.to(device)

    print("Device:", device)
    print("Model:", net)

    params = {key: [np.product(param.size()) for param in child.parameters()] for key, child in net.named_children()}
    n_params = sum(sum(p) for p in params.values())
    print("Parameters:", n_params, params)
    print()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    t = tqdm.trange(n_epochs, unit='epoch')

    metrics = NetMetrics(t, metrics_path, tboard_dir)
    metrics.add_graph(net, device, train_loader)

    for epoch in t:
        train_loss = 0

        for batch in train_loader:
            inputs, targets = (tensor.to(device) for tensor in batch)

            # Reset all the stored gradients
            optimizer.zero_grad()

            # Perform forward, backward and optimization steps
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss

        train_loss /= len(train_loader)
        valid_loss = validate_net(net, criterion, valid_loader, device)

        metrics.update(epoch, train_loss, valid_loss)

    save_network(model_dir, net)
    metrics.finalize(metrics_path)


def validate_net(net, criterion, valid_loader, device):
    """

    :param net:
    :param criterion:
    :param valid_loader:
    :param device:
    :return:
    """
    with torch.no_grad():
        valid_loss = 0

        for batch in valid_loader:
            inputs, targets = (tensor.to(device) for tensor in batch)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss

        # FIXME
        valid_loss /= len(valid_loader)

    return valid_loss
