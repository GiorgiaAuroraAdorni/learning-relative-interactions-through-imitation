import xarray as xr


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
        return xr.Dataset(self.data, coords=self.coords, attrs=self.attrs)

    class Template:
        def __init__(self, **kwargs):
            self.template = {}
            self.update(**kwargs)

        def update(self, **kwargs):
            self.template.update(kwargs)
