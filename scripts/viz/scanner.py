import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtCore import QObject


class DistanceScannerViz(QObject):
    def __init__(self, marxbot, sensor_range=150.0, refresh_interval=30):
        super().__init__()

        self.marxbot = marxbot

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, polar=True)

        self.angles = np.linspace(-np.pi, np.pi, 180)
        distances = np.full_like(self.angles, sensor_range)

        self.plot = self.ax.plot(self.angles, distances, "-k", zorder=1)[0]
        self.scatter = self.ax.scatter(self.angles, distances, marker=".", zorder=2)

        self.ax.fill_between(self.angles, self.marxbot.radius, color=[0.0, 0.0, 1.0, 0.6])

        self.startTimer(refresh_interval)

    def timerEvent(self, event):
        distances = self.marxbot.scanner_distances
        colors = np.array(self.marxbot.scanner_image)

        self.plot.set_ydata(distances)

        offsets = np.stack([self.angles, distances], axis=-1)
        self.scatter.set_offsets(offsets)
        self.scatter.set_facecolors(colors)

        self.fig.canvas.draw_idle()
