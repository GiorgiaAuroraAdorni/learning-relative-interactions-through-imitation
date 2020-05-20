import itertools
import numpy as np
import matplotlib.pyplot as plt

from viz.env import Viz, Env
from plots import draw_docking_station, draw_marxbot


class TrajectoryViz(Viz):
    def __init__(self, *marxbots, time_window=20, show_goals=True, colours=None, goal_colours=None):
        """
        Visualization that shows the trajectories of one or more marXbots in the
        world.

        :param marxbots: The Source objects for the robots to visualize
        :param time_window: Show the trajectories for the last time_window seconds.
        :param show_goals: Whether to show the goal positions, possible values: True, False, 'first' [default True]
        :param colours: The colours to use for the marXbots' current positions
        :param goal_colours: The colours to use for the marXbots' goal positions
        """
        super().__init__()

        self.trajectories = []

        if colours is None:
            colours = ['tab:blue', 'tab:purple']

        if goal_colours is None:
            goal_colours = itertools.cycle(['tab:orange'])

        for i, (marxbot, colour, goal_colour) in enumerate(zip(marxbots, colours, goal_colours)):
            show_goal = (show_goals is True) or (show_goals == 'first' and i == 0)
            trajectory = _SingleTrajectoryViz(marxbot, time_window, show_goal, colour, goal_colour)

            self.trajectories.append(trajectory)

    def _show(self, env: Env):
        self.ax = env.get_axes()

        for trajectory in self.trajectories:
            trajectory.show(self.ax, env.refresh_interval)

        draw_docking_station(self.ax)

        self.ax.set_title('Trajectories')
        self.ax.set_xlabel("x axis")
        self.ax.set_ylabel("y axis")

        self.ax.set_xlim(-250, 250)
        self.ax.set_ylim(-220, 220)
        self.ax.set_aspect('equal')

        self.ax.grid(True)
        self.ax.set_axisbelow(True)

    def _update(self):
        for trajectory in self.trajectories:
            trajectory.update()


class _SingleTrajectoryViz:
    def __init__(self, marxbot, time_window, show_goal, colour, goal_colour):
        self.marxbot = marxbot
        self.time_window = time_window
        self.show_goal = show_goal
        self.colour = colour
        self.goal_colour = goal_colour

    def show(self, ax: plt.Axes, refresh_interval):
        self.n_samples = round(self.time_window / refresh_interval)
        self.trajectory = np.full([2, self.n_samples], np.nan)

        self.plot: plt.Line2D = ax.plot(
            self.trajectory[0], self.trajectory[1],
            color=self.colour, linewidth=1
        )[0]

        if self.show_goal:
            self.goal_pose = draw_marxbot(ax, label="goal position", colour=self.goal_colour)

        self.current_pose = draw_marxbot(ax, label="current position", colour=self.colour)

    def update(self):
        self.trajectory = np.roll(self.trajectory, -1, axis=1)
        self.trajectory[:, -1] = self.marxbot.position

        self.plot.set_data(self.trajectory)

        if self.show_goal:
            self.goal_pose.update(self.marxbot.goal_position, self.marxbot.goal_angle)

        self.current_pose.update(self.marxbot.position, self.marxbot.angle)
