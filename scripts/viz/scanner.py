import numpy as np
from matplotlib import colors, patches
from matplotlib.projections import PolarAxes
from typing import cast

from viz.env import Viz, Env


class LaserScannerViz(Viz):
    def __init__(self, marxbot, sensor_range=150.0):
        self.marxbot = marxbot
        self.marxbot_radius = 8.5
        self.sensor_range = sensor_range

        self.angles = np.linspace(-np.pi, np.pi, 180)
        self.distances = np.full_like(self.angles, self.sensor_range)

    def _show(self, env: Env):
        self.ax = cast(PolarAxes, env.get_axes(polar=True))
        self.ax.set_title('Laser scanner readings', pad=20)

        rmax = self.sensor_range * 1.1
        rticks = np.arange(30, rmax, 30)
        rposition = 135.0
        self.ax.set_rlim(0, rmax)
        self.ax.set_rticks(rticks)
        self.ax.set_rlabel_position(rposition)

        # Set a smaller size to all tick labels
        self.ax.tick_params(labelsize=8)

        # Draw the tick labels perpendicular to the contour lines, centered and
        # with white background for legibility.
        self.ax.tick_params(axis='y', labelrotation=rposition - 90.0)

        for t in self.ax.yaxis.get_major_ticks():
            label = t.label1
            label.set_horizontalalignment('center')
            label.set_verticalalignment('center')
            label.set_bbox({
                'boxstyle': 'round',
                'facecolor': colors.to_rgba('w'),
                'edgecolor': colors.to_rgba('w', alpha=0.0),
            })

        self.plot = self.ax.plot(self.angles, self.distances, "-k", zorder=2)[0]
        self.scatter = self.ax.scatter(self.angles, self.distances, marker=".", fc='k', zorder=3)

        transform = self.ax.transProjectionAffine + self.ax.transAxes
        marxbot = patches.Circle((0, 0), radius=self.marxbot_radius,
                                 edgecolor='tab:blue', facecolor=colors.to_rgba('tab:blue', alpha=0.5),
                                 transform=transform, zorder=3)
        self.ax.add_patch(marxbot)

    def _update(self):
        distances = self.marxbot.scanner_distances
        colors = self.marxbot.scanner_image

        self.plot.set_ydata(distances)

        offsets = np.stack([self.angles, distances], axis=-1)
        self.scatter.set_offsets(offsets)
        self.scatter.set_facecolors(colors)
