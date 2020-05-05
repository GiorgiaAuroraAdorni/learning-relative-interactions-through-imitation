import os

import numpy as np
import xarray as xr

from typing import Union, Tuple
from utils import unpack_tuple


class DatasetBuilder:
    def __init__(self, data, coords=None, attrs=None):
        self.data = self._create_storage(data)
        self.coords = self._create_storage(coords or {})
        self.attrs = attrs or {}

    def _create_storage(self, specs):
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
        return DatasetBuilder.Template(**kwargs)

    def append_sample(self, sample):
        for key, value in sample.template.items():
            if key in self.data:
                self.data[key][1].append(value)
            elif key in self.coords:
                self.coords[key][1].append(value)
            else:
                raise ValueError("Variable '{}' was not defined at initialization time".format(key))

    def finalize(self) -> xr.Dataset:
        dataset = xr.Dataset(self.data, coords=self.coords, attrs=self.attrs)
        dataset['sample'] = np.arange(dataset.sizes['sample'])

        return dataset

    class Template:
        def __init__(self, **kwargs):
            self.template = {}
            self.update(**kwargs)

        def update(self, **kwargs):
            self.template.update(kwargs)


def load_dataset(runs_dir, name='simulation', *, load_dataset=True, load_splits=False)\
        -> Union[Tuple[xr.Dataset, xr.Dataset], xr.Dataset]:
    out = ()

    if load_dataset:
        dataset_path = os.path.join(runs_dir, '%s.nc' % name)
        dataset = xr.open_dataset(dataset_path)
        out = (*out, dataset)

    if load_splits:
        splits_path = os.path.join(runs_dir, '%s.splits.nc' % name)
        splits = xr.open_dataset(splits_path)

        # xarray saves DataArrays as Datasets when writing netCDF files, convert
        # back to DataArray
        splits = splits.split

        out = (*out, splits)

    return unpack_tuple(out)


def save_dataset(runs_dir, name='simulation', *, dataset=None, splits=None):
    if dataset:
        dataset_path = os.path.join(runs_dir, '%s.nc' % name)

        # TODO: some columns don't seems good candidates for zlib compression,
        #       disabling it for these columns might be beneficial.
        encoding = {key: {'zlib': True, 'complevel': 7} for key in dataset.keys()}

        dataset.to_netcdf(dataset_path, encoding=encoding)

    if splits:
        splits.name = 'split'
        splits_path = os.path.join(runs_dir, '%s.splits.nc' % name)
        splits.to_netcdf(splits_path)


def generate_splits(dataset, coord='run', splits=None):
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
