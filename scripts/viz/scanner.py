import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from PyQt5.QtCore import QObject


class DistanceScannerViz(QObject):
    def __init__(self, marxbot, sensor_range=150.0):
        super().__init__()

        self.marxbot = marxbot

        self.angles = np.linspace(-np.pi, np.pi, 180)
        self.distances = np.full_like(self.angles, sensor_range)

    def show(self, refresh_interval=0.030, ax=None):
        if not ax:
            fig = plt.figure()
            ax  = fig.add_subplot(111, polar=True)

        self.ax = ax

        self.plot = self.ax.plot(self.angles, self.distances, "-k", zorder=1)[0]
        self.scatter = self.ax.scatter(self.angles, self.distances, marker=".", zorder=2)

        self.ax.fill_between(self.angles, self.marxbot.radius, color=colors.to_rgba("b", alpha=0.6))

        self.startTimer(int(1000 * refresh_interval))

    def timerEvent(self, event):
        distances = self.marxbot.scanner_distances
        colors = np.array(self.marxbot.scanner_image)

        self.plot.set_ydata(distances)

        offsets = np.stack([self.angles, distances], axis=-1)
        self.scatter.set_offsets(offsets)
        self.scatter.set_facecolors(colors)
