import matplotlib.pyplot as plt

from viz.env import Viz, Env


class _SubplotEnv(Env):
    def __init__(self, gridspec, env):
        self.gridspec = gridspec
        self.env = env

        self.current_subplot = None

    def set_subplot(self, i):
        self.current_subplot = i

    # Env implementation

    def get_axes(self, *args, **kwargs):
        return self.env.get_axes(*self.gridspec, self.current_subplot, *args, **kwargs)

    @property
    def refresh_interval(self) -> float:
        return self.env.refresh_interval


class GridLayoutViz(Viz):
    def __init__(self, gridspec, vizs):
        self.gridspec = gridspec
        self.vizs = vizs

    def _show(self, env: Env):
        env = _SubplotEnv(self.gridspec, env)

        for i, viz in enumerate(self.vizs):
            env.set_subplot(i + 1)
            viz.show(env)

    def _update(self):
        for viz in self.vizs:
            viz.update()
