import itertools
import numpy as np
import matplotlib.pyplot as plt

import plots
from viz.env import Viz, Env


class TrajectoryViz(Viz):
    def __init__(self, *marxbots, time_window=20, show_goals=True, show_initials=True,
                 colours=None, goal_colours=None, initial_colours=None, goal_object='station', title="Trajectories"):
        """
        Visualization that shows the trajectories of one or more marXbots in the
        world.

        :param marxbots: The Source objects for the robots to visualize
        :param time_window: Show the trajectories for the last time_window seconds.
        :param show_goals: Show the goal positions, possible values: True, False, 'first' [default True]
        :param show_initials: Show the initial positions, possible values: True, False, 'first' [default True]
        :param colours: The colours to use for the marXbots' current positions
        :param goal_colours: The colours to use for the marXbots' goal positions
        :param initial_colours: The colours to use for the marXbots' initial positions
        """
        super().__init__()

        self.trajectories = []

        if colours is None:
            colours = ['tab:blue', 'tab:purple']

        if isinstance(colours, str):
            colours = itertools.cycle([colours])

        if goal_colours is None:
            goal_colours = 'tab:orange'

        if isinstance(goal_colours, str):
            goal_colours = itertools.cycle([goal_colours])

        if initial_colours is None:
            initial_colours = 'tab:cyan'

        if isinstance(initial_colours, str):
            initial_colours = itertools.cycle([initial_colours])

        self.goal_object = goal_object
        self.title = title

        params = zip(marxbots, colours, goal_colours, initial_colours)
        for i, params in enumerate(params):
            marxbot, colour, goal_colour, initial_colour = params

            is_first = (i == 0)
            show_goal = (show_goals is True) or (show_goals == 'first' and is_first)
            show_initial = (show_initials is True) or (show_initials == 'first' and is_first)
            show_labels = is_first

            trajectory = _SingleTrajectoryViz(
                marxbot, time_window,
                show_goal, show_initial, show_labels,
                colour, goal_colour, initial_colour
            )

            self.trajectories.append(trajectory)

    def _show(self, env: Env):
        self.ax = env.get_axes()

        for trajectory in self.trajectories:
            trajectory.show(self.ax, env.refresh_interval)

        plots.draw_docking_station(self.ax, self.goal_object)

        self.ax.set_title(self.title)
        self.ax.set_xlabel("x axis")
        self.ax.set_ylabel("y axis")

        self.ax.set_xlim(-250, 250)
        self.ax.set_ylim(-220, 220)
        self.ax.set_aspect('equal')

        self.ax.grid(True)
        self.ax.set_axisbelow(True)
        self.ax.legend(loc='lower left')

    def _update(self):
        for trajectory in self.trajectories:
            trajectory.update()


class _SingleTrajectoryViz:
    def __init__(self, marxbot, time_window, show_goal, show_initial, show_labels, colour, goal_colour, initial_colour):
        self.marxbot = marxbot
        self.time_window = time_window
        self.show_goal = show_goal
        self.show_initial = show_initial
        self.show_labels = show_labels
        self.colour = colour
        self.goal_colour = goal_colour
        self.initial_colour = initial_colour

    def show(self, ax: plt.Axes, refresh_interval):
        self.n_samples = round(self.time_window / refresh_interval)
        self.trajectory = np.full([2, self.n_samples], np.nan)

        self.plot: plt.Line2D = ax.plot(
            self.trajectory[0], self.trajectory[1],
            color=self.colour, linewidth=1
        )[0]

        if self.show_initial:
            label = "initial" if self.show_labels else None
            self.initial_pose = plots.draw_marxbot(ax, label=label, colour=self.initial_colour)

        label = "current" if self.show_labels else None
        self.current_pose = plots.draw_marxbot(ax, label=label, colour=self.colour)

        if self.show_goal:
            label = "goal" if self.show_labels else None
            self.goal_pose = plots.draw_marxbot(ax, label=label, colour=self.goal_colour, show_range=False)

    def update(self):
        self.trajectory = np.roll(self.trajectory, -1, axis=1)
        self.trajectory[:, -1] = self.marxbot.position

        self.plot.set_data(self.trajectory)

        if self.show_goal:
            self.goal_pose.update(self.marxbot.goal_position, self.marxbot.goal_angle)

        if self.show_initial:
            self.initial_pose.update(self.marxbot.initial_position, self.marxbot.initial_angle)

        self.current_pose.update(self.marxbot.position, self.marxbot.angle)
