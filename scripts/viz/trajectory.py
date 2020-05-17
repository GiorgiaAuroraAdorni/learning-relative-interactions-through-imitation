import numpy as np
from matplotlib import colors, patches
from matplotlib.projections import PolarAxes
import matplotlib.pyplot as plt
from typing import cast

from viz.env import Viz, Env
from plots import draw_docking_station, draw_marxbot


class TrajectoryViz(Viz):
    def __init__(self, marxbot, time_window=20):
        super().__init__()

        self.marxbot = marxbot
        self.time_window = time_window

    def _show(self, env: Env):
        self.ax = env.get_axes()

        self.n_samples = round(self.time_window / env.refresh_interval)
        self.trajectory = np.full([2, self.n_samples], np.nan)

        self.plot: plt.Line2D = self.ax.plot(
            self.trajectory[0], self.trajectory[1],
            linewidth=1
        )[0]

        draw_docking_station(self.ax)
        self.current_pose = draw_marxbot(self.ax, None, None, label="current position")
        self.goal_pose = draw_marxbot(self.ax, None, None, label="goal position")

        self.ax.set_title('Trajectory')
        self.ax.set_xlabel("x axis")
        self.ax.set_ylabel("y axis")

        self.ax.axis('equal')
        self.ax.set_xlim(-200, 200)
        self.ax.set_ylim(-200, 200)

        self.ax.grid(True)

    def _update(self):
        self.trajectory = np.roll(self.trajectory, -1, axis=1)
        self.trajectory[:, -1] = self.marxbot.position

        self.current_pose.update(self.marxbot.position, self.marxbot.angle)
        self.goal_pose.update(self.marxbot.goal_position, self.marxbot.goal_angle)

        self.plot.set_data(self.trajectory)
