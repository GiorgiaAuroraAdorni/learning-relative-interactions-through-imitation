import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, dropout):
        """

        """
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
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(128, 128)
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
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
        net = self.drop1(net)
        net = F.relu(self.fc2(net))
        net = self.drop2(net)
        output = self.fc3(net)

        return output


class ConvNet_MaxPool(nn.Module):
    def __init__(self, dropout):
        """

        """
        super().__init__()

        # Using BatchNorm somewhat improperly as an input normalization step.
        # This layer applies the transformation y = (x - mu) / std using per-channel
        # mean and stdev over the train set. After that, z = alpha * y + beta,
        # where alpha and beta are learned parameters, which allow the model to
        # learn the best scaling for the data.
        self.in_norm = nn.BatchNorm1d(num_features=4)

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=5,
                               stride=2, padding=2, padding_mode='circular')
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=96, kernel_size=5,
                               stride=2, padding=2, padding_mode='circular')
        self.maxp1 = nn.MaxPool1d(kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=5,
                               stride=1, padding=2, padding_mode='circular')

        self.fc1 = nn.Linear(15 * 96, 128)
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(128, 128)
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc3 = nn.Linear(128, 2)

    def forward(self, input):
        """

        :param input: input tensor with shape samples ⨉ channels ⨉ angles
        :return: output tensor with shape samples ⨉ wheels
        """
        net = self.in_norm(input)

        net = F.relu(self.conv1(net))
        net = F.relu(self.conv2(net))

        # Adding the same circular padding as the other layers, using F.pad since
        # nn.MaxPool1d only supports zero padding.
        net = self.maxp1(F.pad(net, [1, 1], mode='circular'))

        net = F.relu(self.conv3(net))

        net = torch.flatten(net, start_dim=1)

        net = F.relu(self.fc1(net))
        net = self.drop1(net)
        net = F.relu(self.fc2(net))
        net = self.drop2(net)
        output = self.fc3(net)

        return output
