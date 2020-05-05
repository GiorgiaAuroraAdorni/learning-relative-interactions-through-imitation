import numpy as np
from matplotlib import colors

from viz.env import Viz


class DistanceScannerViz(Viz):
    def __init__(self, marxbot, sensor_range=150.0):
        self.marxbot = marxbot

        self.angles = np.linspace(-np.pi, np.pi, 180)
        self.distances = np.full_like(self.angles, sensor_range)

    def _show(self, env):
        self.ax = env.get_axes(polar=True)

        self.plot = self.ax.plot(self.angles, self.distances, "-k", zorder=1)[0]
        self.scatter = self.ax.scatter(self.angles, self.distances, marker=".", zorder=2)

        self.ax.fill_between(self.angles, self.marxbot.radius, color=colors.to_rgba("b", alpha=0.6))

    def _update(self):
        distances = self.marxbot.scanner_distances
        colors = np.array(self.marxbot.scanner_image)

        self.plot.set_ydata(distances)

        offsets = np.stack([self.angles, distances], axis=-1)
        self.scatter.set_offsets(offsets)
        self.scatter.set_facecolors(colors)
