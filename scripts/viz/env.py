import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from PyQt5.QtCore import QObject


class Env(ABC):
    @abstractmethod
    def get_axes(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def refresh_interval(self) -> float:
        pass


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


class InteractiveEnv(QObject): # Should inherit from Env
    def __init__(self, vizs, refresh_interval=0.060):
        """
        :param Sequence[Viz] vizs: sequence of visualizations to display
        :param float refresh_interval: refresh interval in ms
        """
        super().__init__()

        self.vizs = vizs
        self._refresh_interval = refresh_interval

    def show(self, **fig_kw):
        self.fig = plt.figure(**fig_kw)

        for viz in self.vizs:
            viz.show(self)

        self.fig.tight_layout()

        self.startTimer(int(1000 * self.refresh_interval))

    def timerEvent(self, event):
        for viz in self.vizs:
            viz.update()

    # Env implementation

    def get_axes(self, *args, **kwargs):
        return self.fig.add_subplot(*args, **kwargs)

    @property
    def refresh_interval(self):
        return self._refresh_interval
