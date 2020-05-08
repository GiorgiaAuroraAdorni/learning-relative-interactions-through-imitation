import os

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from dataset import load_dataset
from geometry import Point, Transform
from utils import unpack
import viz


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
    dataset_states = load_dataset(runs_dir)

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
    dataset_states = load_dataset(runs_dir)

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
    dataset_states = load_dataset(runs_dir)

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


def plot_trajectory(runs_dir, img_dir, title, filename):
    """

    :param runs_dir:
    :param img_dir:
    :param title:
    :param filename:
    """
    dataset_states = load_dataset(runs_dir)

    run_states = dataset_states.where(dataset_states.run == 0, drop=True)

    goal_position = run_states.goal_position[0]
    goal_angle = run_states.goal_angle[0]

    init_position = run_states.initial_position[0]
    init_angle = run_states.initial_angle[0]

    x_position, y_position = unpack(run_states.position, 'axis')

    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    ax.set_xlabel('x axis', fontsize=11)
    ax.set_ylabel('y axis', fontsize=11)

    ax.grid()

    points = Point.from_list([
        Point.ORIGIN,
        [1, -3, 1], [1, 3, 1], [6, 0, 1]
    ])

    origin_tform = Transform.pose_transform(init_position, init_angle)
    origin_points = points.transformed(origin_tform).to_euclidean().T

    goal_tform = Transform.pose_transform(goal_position, goal_angle)
    goal_points = points.transformed(goal_tform).to_euclidean().T

    radius = 8.5
    ax.add_patch(plt.Circle(origin_points[0], radius,
                            facecolor=colors.to_rgba('tab:blue', alpha=0.5),
                            edgecolor='tab:blue', linewidth=1.5,
                            label='initial position'))
    ax.add_patch(plt.Polygon(origin_points[1:],
                             facecolor=colors.to_rgba('tab:blue', alpha=0),
                             edgecolor='tab:blue'))

    ax.add_patch(plt.Circle(goal_points[0], radius,
                            facecolor=colors.to_rgba('tab:orange', alpha=0.5),
                            edgecolor='tab:orange', linewidth=1.5,
                            label='goal position'))
    ax.add_patch(plt.Polygon(goal_points[1:],
                             facecolor=colors.to_rgba('tab:orange', alpha=0),
                             edgecolor='tab:orange'))

    plt.plot(x_position, y_position, color='black', label='trajectory')

    ax.set_ylim(-init_position[1] + 20, init_position[1] + 20)
    ax.set_xlim(0, init_position[0] * 2)
    ax.set_aspect('equal')

    plt.legend()
    fig.suptitle(title, fontsize=14, weight='bold')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)

    save_visualisation(filename, img_dir, make_space=True, axes=ax)


def plot_sensors(runs_dir, video_dir, title, filename):
    """

    :param runs_dir:
    :param video_dir:
    :param title:
    :param filename:
    """
    dataset_states = load_dataset(runs_dir)
    run_states = dataset_states.where(dataset_states.run == 0, drop=True)

    marxbot = viz.DatasetSource(run_states)

    # Create the visualizations
    env = viz.FuncAnimationEnv([
        viz.GridLayout((1, 2), [
            viz.LaserScannerViz(marxbot),
            viz.ControlSignalsViz(marxbot)
        ], suptitle=title)
    ], sources=[marxbot])
    env.show(figsize=(9, 4))

    video_path = os.path.join(video_dir, '%s.mp4' % filename)
    env.save(video_path, dpi=300)


def plot_initial_positions(runs_dir, img_dir, title, filename):
    dataset_states = load_dataset(runs_dir)
    step_states = dataset_states.where(dataset_states.step == 0, drop=True)
    x, y = unpack(step_states.initial_position, 'axis')

    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    ax.scatter(x, y, alpha=0.2)
    ax.axis('equal')

    # FIXME add labels
    fig.suptitle(title, fontsize=14, weight='bold')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)

    save_visualisation(filename, img_dir, make_space=True, axes=ax)
