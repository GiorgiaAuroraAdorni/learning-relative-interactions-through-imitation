import os


def check_dir(directory):
    """
    Check if the path is a directory, if not create it.
    :param directory: path to the directory
    """
    os.makedirs(directory, exist_ok=True)


def unpack(dataset, dim):
    """
    Unpack a `xr.Dataset` along a given dimension

    :param dataset: dataset object to be unpacked
    :param dim: dimension along which the dataset should be unpacked
    :return: sequence of one `xr.Dataset`s for each value of dim
    """
    return (dataset.loc[{dim: value}] for value in dataset[dim])


def unpack_tuple(x):
    """

    :param x:
    :return:
    """
    if len(x) == 1:
        return x[0]
    else:
        return x
