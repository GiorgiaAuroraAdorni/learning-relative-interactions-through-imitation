import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import get_pos_sensing_control


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
    pickle_file = os.path.join(runs_dir, 'simulation.pkl.gz')
    dataset_states = pd.read_pickle(pickle_file)

    time_steps = np.arange(dataset_states['step'].max() + 1)

    fig, axes = plt.subplots(nrows=2, figsize=(6.8, 8.4), sharex=True)
    plt.xlabel('timestep', fontsize=11)

    # Plot position distance from goal
    p_q1 = dataset_states.groupby(['step'])['goal_position_distance'].quantile(0.25)
    p_q2 = dataset_states.groupby(['step'])['goal_position_distance'].quantile(0.75)
    p_q3 = dataset_states.groupby(['step'])['goal_position_distance'].quantile(0.10)
    p_q4 = dataset_states.groupby(['step'])['goal_position_distance'].quantile(0.90)
    median_p_by_step = dataset_states.groupby(['step'])['goal_position_distance'].median()

    axes[0].set_ylabel('distance from goal', fontsize=11)
    axes[0].grid()

    ln, = axes[0].plot(time_steps, median_p_by_step, label='median')
    axes[0].fill_between(time_steps, p_q1, p_q2, alpha=0.2, label='interquartile range', color=ln.get_color())
    axes[0].fill_between(time_steps, p_q3, p_q4, alpha=0.1, label='interdecile range', color=ln.get_color())

    axes[0].legend()
    axes[0].set_title('Position', weight='bold', fontsize=12)

    # Plot angle distance form goal
    a_q1 = dataset_states.groupby(['step'])['goal_angle_distance'].quantile(0.25)
    a_q2 = dataset_states.groupby(['step'])['goal_angle_distance'].quantile(0.75)
    a_q3 = dataset_states.groupby(['step'])['goal_angle_distance'].quantile(0.10)
    a_q4 = dataset_states.groupby(['step'])['goal_angle_distance'].quantile(0.90)
    median_a_by_step = dataset_states.groupby(['step'])['goal_angle_distance'].median()

    axes[1].set_ylabel('distance from goal', fontsize=11)
    axes[1].grid()

    ln, = axes[1].plot(time_steps, median_a_by_step, label='median')
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
    pickle_file = os.path.join(runs_dir, 'simulation.pkl.gz')
    dataset_states = pd.read_pickle(pickle_file)

    time_steps = np.arange(dataset_states['step'].max() + 1)

    x_goal_position = dataset_states['goal_position'][0][0]
    y_goal_position = dataset_states['goal_position'][0][1]

    df = dataset_states[['position', 'goal_position', 'step']]
    df[['x_position', 'y_position']] = pd.DataFrame(df['position'].tolist(), index=df.index)

    x_mean = df.groupby(['step'])['x_position'].median()
    x_q1 = df.groupby(['step'])['x_position'].quantile(0.25)
    x_q2 = df.groupby(['step'])['x_position'].quantile(0.75)
    x_q3 = df.groupby(['step'])['x_position'].quantile(0.10)
    x_q4 = df.groupby(['step'])['x_position'].quantile(0.90)

    y_mean = df.groupby(['step'])['y_position'].median()
    y_q1 = df.groupby(['step'])['y_position'].quantile(0.25)
    y_q2 = df.groupby(['step'])['y_position'].quantile(0.75)
    y_q3 = df.groupby(['step'])['y_position'].quantile(0.10)
    y_q4 = df.groupby(['step'])['y_position'].quantile(0.90)

    # Plot the evolution of the position over time
    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    ax.set_xlabel('timestep', fontsize=11)
    ax.set_ylabel('position', fontsize=11)
    ax.set_yticks([x_goal_position, y_goal_position])
    ax.grid()

    ln, = ax.plot(time_steps, x_mean, label='median (x axis)')
    ax.fill_between(time_steps, x_q1, x_q2, alpha=0.2, label='interquartile range (x axis)', color=ln.get_color())
    ax.fill_between(time_steps, x_q3, x_q4, alpha=0.1, label='interdecile range (x axis)', color=ln.get_color())

    ln, = ax.plot(time_steps, y_mean, label='median (y axis)')
    ax.fill_between(time_steps, y_q1, y_q2, alpha=0.2, label='interquartile range (y axis)', color=ln.get_color())
    ax.fill_between(time_steps, y_q3, y_q4, alpha=0.1, label='interdecile range (y axis)', color=ln.get_color())

    ax.legend()
    fig.suptitle(title, fontsize=14, weight='bold')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)

    save_visualisation(filename, img_dir, make_space=True, axes=ax)



    plt.plot(time_steps, x_mean, label='x mean')
    plt.fill_between(time_steps, x_mean - x_std, x_mean + x_std, alpha=0.2, label='x +/- 1 std')

    plt.plot(time_steps, y_mean, label='y mean')
    plt.fill_between(time_steps, y_mean - y_std, y_mean + y_std, alpha=0.2, label='y +/- 1 std')

    plt.legend()
    plt.title(title, weight='bold', fontsize=14)

    save_visualisation(filename, img_dir)
