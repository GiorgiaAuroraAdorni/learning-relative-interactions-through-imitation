import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
import xarray as xr

from utils import unpack


def save_visualisation(filename, img_dir, make_space=False, axes=None):
    """

    :param filename:
    :param img_dir:
    :param make_space:
    :param axes:

    """
    file = os.path.join(img_dir, '%s.pdf' % filename)
    img = os.path.join(img_dir, '%s.png' % filename)
    if make_space:
        make_space_above(axes, topmargin=1)

    plt.savefig(file)
    plt.savefig(img)
    plt.close()


def make_space_above(axes, topmargin=1):
    """
    Increase figure size to make topmargin (in inches) space for titles, without changing the axes sizes
    :param axes:
    :param topmargin:
    """
    if type(axes) is np.ndarray:
        fig = axes.flatten()[0].figure
    else:
        fig = axes.figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    fig_h = h - (1 - s.top) * h + topmargin
    fig.subplots_adjust(bottom=s.bottom * h / fig_h, top=1 - topmargin / fig_h)
    fig.set_figheight(fig_h)


def plot_distance_from_goal(runs_dir, img_dir, title, filename):
    """
    :param runs_dir:
    :param img_dir:
    :param title
    :param filename
    """
    nc_file = os.path.join(runs_dir, 'simulation.nc')
    dataset_states = xr.load_dataset(nc_file)

    time_steps = np.arange(dataset_states.step.max() + 1)

    fig, axes = plt.subplots(nrows=2, figsize=(6.8, 8.4), sharex=True)
    plt.xlabel('timestep', fontsize=11)

    # Plot position distance from goal
    goal_p_dist_by_step = dataset_states.goal_position_distance.groupby('step')
    p_q1, p_q2, p_q3, p_q4, p_median = unpack(goal_p_dist_by_step.quantile([0.25, 0.75, 0.10, 0.90, 0.5]), 'quantile')

    axes[0].set_ylabel('distance from goal', fontsize=11)
    axes[0].grid()

    ln, = axes[0].plot(time_steps, p_median, label='median')
    axes[0].fill_between(time_steps, p_q1, p_q2, alpha=0.2, label='interquartile range', color=ln.get_color())
    axes[0].fill_between(time_steps, p_q3, p_q4, alpha=0.1, label='interdecile range', color=ln.get_color())

    axes[0].legend()
    axes[0].set_title('Position', weight='bold', fontsize=12)

    # Plot angle distance form goal
    goal_a_dist_by_step = dataset_states.goal_angle_distance.groupby('step')
    a_q1, a_q2, a_q3, a_q4, a_median = unpack(goal_a_dist_by_step.quantile([0.25, 0.75, 0.10, 0.90, 0.5]), 'quantile')

    axes[1].set_ylabel('distance from goal', fontsize=11)
    axes[1].grid()

    ln, = axes[1].plot(time_steps, a_median, label='median')
    # plt.plot(time_steps, mean_a_diff_from_goal, label='mean')
    axes[1].fill_between(time_steps, a_q1, a_q2, alpha=0.2, label='interquartile range', color=ln.get_color())
    axes[1].fill_between(time_steps, a_q3, a_q4, alpha=0.1, label='interdecile range', color=ln.get_color())

    axes[1].legend()
    axes[1].set_title('Angle', weight='bold', fontsize=12)

    fig.suptitle(title, fontsize=14, weight='bold')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)

    save_visualisation(filename, img_dir, make_space=True, axes=axes)


def plot_position_over_time(runs_dir, img_dir, title, filename):
    """

    :param runs_dir:
    :param img_dir:
    :param title
    :param filename:
    """
    nc_file = os.path.join(runs_dir, 'simulation.nc')
    dataset_states = xr.load_dataset(nc_file)

    time_steps = np.arange(dataset_states.step.max() + 1)
    x_goal_position, y_goal_position = unpack(dataset_states.goal_position[0], 'axis')
    position_by_step = dataset_states.position.groupby('step').quantile([0.25, 0.75, 0.10, 0.90, 0.5])

    # Plot the evolution of the position over time
    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    ax.set_xlabel('timestep', fontsize=11)
    ax.set_ylabel('position', fontsize=11)
    ax.set_yticks([x_goal_position, y_goal_position])
    ax.grid()

    for quantiles in unpack(position_by_step, "axis"):
        axis = quantiles.axis.values
        q1, q2, q3, q4, median = unpack(quantiles, "quantile")

        ln, = ax.plot(time_steps, median, label='median (%s axis)' % axis)
        ax.fill_between(time_steps, q1, q2, alpha=0.2, label='interquartile range (%s axis)' % axis, color=ln.get_color())
        ax.fill_between(time_steps, q3, q4, alpha=0.1, label='interdecile range (%s axis)' % axis, color=ln.get_color())

    ax.legend()
    fig.suptitle(title, fontsize=14, weight='bold')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)

    save_visualisation(filename, img_dir, make_space=True, axes=ax)


def plot_goal_reached_distribution(runs_dir, img_dir, title, filename):
    """

    :param runs_dir:
    :param img_dir:
    :param title:
    :param filename:
    """
    nc_file = os.path.join(runs_dir, 'simulation.nc')
    dataset_states = xr.load_dataset(nc_file)

    time_steps = np.arange(dataset_states.step.max() + 1)

    states_subset = dataset_states[["step", "goal_reached"]]
    last_steps = states_subset.groupby("run").map(lambda x: x.isel(sample=-1))
    [false_label, false_samples], [true_label, true_samples] = last_steps.groupby('goal_reached')

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    plt.hist([true_samples.step, false_samples.step], bins=time_steps, label=[true_label, false_label], stacked=True,
             alpha=0.9)
    plt.ylim(0, 35)
    plt.legend()

    ax.set_xlim(0, dataset_states.step.max() + 1)
    ax.set_xlabel('timestep', fontsize=11)
    ax.set_ylabel('samples', fontsize=11)
    fig.suptitle(title, fontsize=14, weight='bold')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)

    save_visualisation(filename, img_dir, make_space=True, axes=ax)


def plot_sensors(runs_dir, img_dir, title, filename):
    """

    :param runs_dir:
    :param img_dir:
    :param title:
    :param filename:
    """
    pickle_file = os.path.join(runs_dir, 'simulation.pkl.gz')
    dataset_states = pd.read_pickle(pickle_file)

    run_states = dataset_states.loc[dataset_states['run'] == 0]
    angles = np.linspace(-np.pi, np.pi, 180)
    robot_radius = 8.5
    sensor_range = 150.0
    distances = np.full_like(angles, sensor_range)

    yticks = np.arange(50, sensor_range * 2, 50)

    fig = plt.figure()
    fig.suptitle(title, fontsize=12, weight='bold')
    ax = fig.add_subplot(111, polar=True)
    ax.set_yticks(yticks)
    ax.set_ylim(0, 300)
    ax.tick_params(labelsize=8)

    scatter = ax.scatter(angles, distances, marker=".", zorder=2)
    ln, = ax.plot(angles, distances, "-k", zorder=1)
    ax.fill_between(angles, robot_radius, color=[0.0, 0.0, 1.0, 0.6])
    make_space_above(ax, topmargin=1)

    def update(i):
        distances = run_states.loc[i, 'scanner_distances']
        colors = run_states.loc[i, 'scanner_image']

        ln.set_ydata(distances)

        offsets = np.stack([angles, distances], axis=-1)
        scatter.set_offsets(offsets)
        scatter.set_facecolors(colors)

    ani = FuncAnimation(fig, update, frames=run_states.index, blit=False)

    video = os.path.join(img_dir, '%s.mp4' % filename)

    ani.save(video, dpi=300)

    plt.close()
