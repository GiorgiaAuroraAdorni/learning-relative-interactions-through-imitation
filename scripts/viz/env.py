from typing import Optional

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from abc import ABC, abstractmethod


class Env(ABC):
    @abstractmethod
    def get_figure(self) -> plt.Figure:
        pass

    @abstractmethod
    def get_axes(self, *args, **kwargs) -> plt.Axes:
        pass

    @property
    @abstractmethod
    def refresh_interval(self) -> float:
        pass


class PassthroughEnv(Env):
    def __init__(self, env: Env):
        self.env = env

    def get_figure(self):
        return self.env.get_figure()

    def get_axes(self, *args, **kwargs):
        return self.env.get_axes(*args, **kwargs)

    @property
    def refresh_interval(self) -> float:
        return self.env.refresh_interval


class Viz(ABC):
    @abstractmethod
    def _show(self, env: Env):
        pass

    @abstractmethod
    def _update(self):
        pass

    def show(self, env: Env):
        self._show(env)

    def update(self):
        self._update()


class FuncAnimationEnv(Env):
    def __init__(self, vizs, datasets=None, refresh_interval=0.060):
        self.vizs = vizs
        self.datasets = datasets or []

        self._frames = None
        self._refresh_interval = refresh_interval

        self.fig: Optional[plt.Figure] = None
        self.anim: Optional[FuncAnimation] = None

    def show(self, **fig_kw):
        self.fig = plt.figure(constrained_layout=True, **fig_kw)

        for viz in self.vizs:
            viz.show(self)

        self.fig.execute_constrained_layout()
        self.fig.set_constrained_layout(False)

        # Use the minimum length of the datasets to control the number of frames
        # to be drawn. Defaults to None, which results in an infinite animation.
        self._frames = None
        if len(self.datasets) > 0:
            self._frames = min(len(ds) for ds in self.datasets)

        interval = round(1000 * self.refresh_interval)

        self.anim = FuncAnimation(
            self.fig, self.update, frames=self._frames, interval=interval
        )

    def update(self, frame):
        for dataset in self.datasets:
            dataset.update(frame)

        for viz in self.vizs:
            viz.update()

    def save(self, filename, **save_kw):
        t = tqdm(total=self._frames, unit='frames')
        last_frame = 0

        def progress(current_frame, _total_frames):
            nonlocal t, last_frame

            t.update(current_frame - last_frame)
            last_frame = current_frame

        self.anim.save(filename, progress_callback=progress, **save_kw)

        t.update(self._frames - last_frame)

    # Env implementation

    def get_figure(self):
        return self.fig

    def get_axes(self, *args, **kwargs):
        return self.fig.add_subplot(*args, **kwargs)

    @property
    def refresh_interval(self):
        return self._refresh_interval


class AnimationDataset:
    def __init__(self, dataset: xr.Dataset, dim="sample"):
        self.dataset = dataset
        self.dim = dim

        self.current_sample = None

    def update(self, frame):
        self.current_sample = self.dataset[{self.dim: frame}]

    def __len__(self):
        return self.dataset.sizes[self.dim]

    def __getattr__(self, name):
        return getattr(self.current_sample, name)
