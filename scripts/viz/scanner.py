import numpy as np
from matplotlib import colors

from viz.env import Viz


class DistanceScannerViz(Viz):
    def __init__(self, marxbot, sensor_range=150.0):
        self.marxbot = marxbot
        self.marxbot_radius = 8.5
        self.sensor_range = sensor_range

        self.angles = np.linspace(-np.pi, np.pi, 180)
        self.distances = np.full_like(self.angles, self.sensor_range)

    def _show(self, env):
        self.ax = env.get_axes(polar=True)
        # self.ax.set_title('Laser scanner response over time', weight='bold', fontsize=12)

        yticks = np.arange(50, self.sensor_range * 1.1, 50)
        self.ax.set_yticks(yticks)
        self.ax.set_ylim(0, self.sensor_range * 1.1)
        self.ax.tick_params(labelsize=8)

        self.plot = self.ax.plot(self.angles, self.distances, "-k", zorder=1)[0]
        self.ax.text(0.5, 0.5, 'Laser scanner response over time', ha='left', va='center')
        self.scatter = self.ax.scatter(self.angles, self.distances, marker=".", fc='k', zorder=2)

        self.ax.fill_between(self.angles, self.marxbot_radius, color=colors.to_rgba("b", alpha=0.6))

    def _update(self):
        distances = self.marxbot.scanner_distances
        colors = self.marxbot.scanner_image

        self.plot.set_ydata(distances)

        offsets = np.stack([self.angles, distances], axis=-1)
        self.scatter.set_offsets(offsets)
        self.scatter.set_facecolors(colors)
