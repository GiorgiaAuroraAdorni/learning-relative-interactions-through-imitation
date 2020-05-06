import os
from typing import Union, Tuple

import numpy as np
import xarray as xr

from utils import unpack_tuple


class DatasetBuilder:
    """
    TODO: add documentation please
    """
    def __init__(self, data, coords=None, attrs=None):
        self.data = self._create_storage(data)
        self.coords = self._create_storage(coords or {})
        self.attrs = attrs or {}

    def _create_storage(self, specs):
        """

        :param specs:
        :return storage:
        """
        storage = dict()

        for key, value in specs.items():
            data = []
            attrs = {}

            if isinstance(value, str):
                # A single dimension name: ("dimension")
                dims = (value,)
            elif isinstance(value, tuple):
                if len(value) in [2, 3] and (value[0] == Ellipsis or value[0] ==(Ellipsis,)):
                    # Definition of a dimension coordinate:
                    #   (..., ["x", "y"])
                    #   (..., ["x", "y"], {"attr": "value"})
                    dims = Ellipsis
                    data = value[1]

                    if len(value) == 3:
                        attrs = value[2]
                elif len(value) == 2 and isinstance(value[1], dict):
                    # A sequence of dimensions and a dict of attributes:
                    #   (("dim1", "dim2"), {"attr": "value"})
                    dims, attrs = value
                else:
                    # A sequence of dimensions
                    dims = value
            else:
                raise ValueError("Unexpected format in definition of variable '{}'".format(key))

            if dims == Ellipsis:
                # Dimension coordinate, the only dimension is the coordinate name
                dims = (key,)
            else:
                # Add an implicit sample dimension
                dims = ("sample", *dims)

            # Create the tuple in the format expected by xr.Dataset
            storage[key] = (dims, data, attrs)

        return storage

    def create_template(self, **kwargs):
        """

        :param kwargs:
        :return DatasetBuilder.Template:
        """
        return DatasetBuilder.Template(**kwargs)

    def append_sample(self, sample):
        """

        :param sample:
        """
        for key, value in sample.template.items():
            if key in self.data:
                self.data[key][1].append(value)
            elif key in self.coords:
                self.coords[key][1].append(value)
            else:
                raise ValueError("Variable '{}' was not defined at initialization time".format(key))

    def finalize(self) -> xr.Dataset:
        """

        :return dataset:
        """
        dataset = xr.Dataset(self.data, coords=self.coords, attrs=self.attrs)
        dataset['sample'] = np.arange(dataset.sizes['sample'])

        return dataset

    class Template:
        """

        """
        def __init__(self, **kwargs):
            self.template = {}
            self.update(**kwargs)

        def update(self, **kwargs):
            """

            :param kwargs:
            :return:
            """
            self.template.update(kwargs)


def load_dataset(runs_dir, name='simulation', *, load_dataset=True, load_splits=False)\
        -> Union[Tuple[xr.Dataset, xr.Dataset], xr.Dataset]:
    """

    :param runs_dir:
    :param name:
    :param load_dataset:
    :param load_splits:
    :return:
    """
    out = ()

    if load_dataset:
        dataset_path = os.path.join(runs_dir, '%s.nc' % name)
        dataset = xr.open_dataset(dataset_path)

        # Support loading old versions of the datasets, by migrating them to the
        # current version
        dataset = migrate_dataset(dataset)

        out = (*out, dataset)

    if load_splits:
        splits_path = os.path.join(runs_dir, '%s.splits.nc' % name)
        splits = xr.open_dataset(splits_path)

        # xarray saves DataArrays as Datasets when writing netCDF files, convert
        # back to DataArray
        splits = splits.split

        out = (*out, splits)

    return unpack_tuple(out)


# Current version number of dataset files
CURRENT_VERSION = 2


def migrate_dataset(dataset):
    """
        Support loading old versions of the datasets, by migrating them to the
        current version.

        Every time a breaking change is made to the dataset format, the current
        version number should be increased and a migration that transforms a
        dataset from the old format to the new one should be added to this function.
    :param dataset: a dataset in a potentially old format
    :return: dataset converted to the latest format
    """
    version = dataset.attrs.get('version', 1)

    if version < 2:
        # Migrate from v1 to v2:
        #  * Rename wheel_target_speed to wheel_target_speeds
        dataset = dataset.rename(wheel_target_speed='wheel_target_speeds')

    # Set the current version
    dataset.attrs['version'] = CURRENT_VERSION

    return dataset


def save_dataset(runs_dir, name='simulation', *, dataset=None, splits=None):
    """

    :param runs_dir:
    :param name:
    :param dataset:
    :param splits:
    """
    if dataset:
        dataset_path = os.path.join(runs_dir, '%s.nc' % name)

        # TODO: some columns don't seems good candidates for zlib compression,
        #       disabling it for these columns might be beneficial.
        encoding = {key: {'zlib': True, 'complevel': 7} for key in dataset.keys()}

        # Set the current version
        dataset.attrs['version'] = CURRENT_VERSION

        dataset.to_netcdf(dataset_path, encoding=encoding)

    if splits:
        splits.name = 'split'
        splits_path = os.path.join(runs_dir, '%s.splits.nc' % name)
        splits.to_netcdf(splits_path)


def generate_splits(dataset, coord='run', splits=None):
    """

    :param dataset:
    :param coord:
    :param splits:
    :return splits:
    """
    if splits is None:
        splits = {
            "train": 0.7,
            "validation": 0.15,
            "test": 0.15
        }

    names = list(splits.keys())
    codes = np.arange(len(splits), dtype=np.int8)
    probs = list(splits.values())

    unique, unique_inverse = np.unique(dataset[coord], return_inverse=True)
    n_indices = unique.size

    unique_assigns = np.random.choice(codes, n_indices, p=probs)
    assigns = unique_assigns[unique_inverse]

    splits = xr.DataArray(assigns, coords=dataset[coord].coords, attrs={"split_names": names})

    return splits
