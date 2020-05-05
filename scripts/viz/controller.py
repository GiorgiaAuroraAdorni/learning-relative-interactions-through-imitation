import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtCore import QObject

from marxbot import MyMarxbot


class ControllerViz(QObject):
    def __init__(self, marxbot: MyMarxbot, time_window=10):
        super().__init__()

        self.marxbot = marxbot
        self.controller = marxbot.controller

        self.time_window = time_window

    def show(self, refresh_interval=0.060, ax=None):
        if not ax:
            fig, ax = plt.subplots()

        self.ax = ax

        self.ax.set_xlabel("time (s)")
        self.ax.set_xlim(-self.time_window, 0)

        self.n_samples = round(self.time_window / refresh_interval)
        self.n_dims = 2
        self.time = np.linspace(-self.time_window, 0, self.n_samples)
        self.readings = np.full((self.n_dims, self.n_samples), np.nan)

        labels = ["linear velocity", "angular velocity"]
        colors = ["tab:blue", "tab:orange"]
        mins = [-self.controller.max_vel, -np.pi / 4 * self.controller.max_vel]
        maxs = [+self.controller.max_vel, +np.pi / 4 * self.controller.max_vel]

        self.plots = []

        for i in range(self.n_dims):
            ax = self.ax

            if i > 0:
                ax = ax.twinx()

            ax.set_ylabel(labels[i], color=colors[i])
            ax.tick_params(axis='y', labelcolor=colors[i])

            plot = ax.plot(self.time, self.readings[i], color=colors[i])[0]

            ax.set_ylim(
                mins[i] - 0.1 * abs(mins[i]),
                maxs[i] + 0.1 * abs(maxs[i])
            )

            self.plots.append(plot)

        self.startTimer(int(1000 * refresh_interval))

    def timerEvent(self, event):
        self.readings = np.roll(self.readings, -1, axis=1)
        self.readings[:, -1] = (self.controller.lin_vel, self.controller.ang_vel)

        for i in range(self.n_dims):
            self.plots[i].set_ydata(self.readings[i])
