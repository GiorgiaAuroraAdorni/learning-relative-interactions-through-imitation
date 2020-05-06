import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

import xarray as xr
import tqdm
import math


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


def split_datasets(dataset, splits):
    # Force-load the datasets from disk before splitting them, since it's _very_
    # much faster than reading them in random order once they've been split.
    dataset.load()

    # Attach the split column to the dataset, containing the id of the split
    # each sample belongs to
    dataset['split'] = splits

    return (split for _, split in dataset.groupby('split'))


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


def train_net(dataset, splits, n_epochs=100, lr=0.01, batch_size=1024):
    train, validation, _ = split_datasets(dataset, splits)

    train_loader = to_torch_loader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = to_torch_loader(validation, batch_size=batch_size, shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = ConvNet()
    net.to(device, non_blocking=True)

    print("Device:", device)
    print("Model:", net)

    n_params = list(math.prod(param.size()) for param in net.parameters())
    print("Parameters:", n_params)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    t = tqdm.trange(n_epochs, unit='epoch')

    for epoch in t:
        l = 0

        for batch in train_loader:
            inputs, targets = (tensor.to(device) for tensor in batch)

            # Reset all the stored gradients
            optimizer.zero_grad()

            # Perform forward, backward and optimization steps
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            l += loss

        l /= len(train_loader) * batch_size

        vmetrics = validate_net(net, criterion, valid_loader, device)

        metrics = {
            "t. loss": float(l)
        }
        metrics.update(vmetrics)

        t.set_postfix(metrics)

    model_path = 'datasets/prova.pth'
    torch.save(net.state_dict(), model_path)


def validate_net(net, criterion, valid_loader, device):
    with torch.no_grad():
        l = 0

        for batch in valid_loader:
            inputs, targets = (tensor.to(device) for tensor in batch)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            l += loss

        # FIXME
        l /= len(valid_loader) * 1024

    metrics = {
        "v. loss": float(l)
    }

    return metrics
