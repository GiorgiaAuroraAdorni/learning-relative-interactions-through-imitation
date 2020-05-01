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
    fig = axes.flatten()[0].figure
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

    mean_by_step = dataset_states.groupby(['step']).mean()
    std_by_step = dataset_states.groupby(['step']).std()

    mean_p_dist_from_goal = mean_by_step['goal_position_distance']
    std_p_dist_from_goal = std_by_step['goal_position_distance']

    mean_a_diff_from_goal = mean_by_step['goal_angle_distance']
    std_a_diff_from_goal = std_by_step['goal_angle_distance']

    plt.figure()
    fig, axes = plt.subplots(nrows=2, figsize=(7, 10), sharex=True)
    plt.xlabel('timestep', fontsize=11)

    # Plot position distance from goal
    axes[0].set_ylabel('distance from goal', fontsize=11)
    axes[0].grid()

    axes[0].plot(time_steps, mean_p_dist_from_goal, label='mean')
    axes[0].fill_between(time_steps,
                         mean_p_dist_from_goal - std_p_dist_from_goal,
                         mean_p_dist_from_goal + std_p_dist_from_goal,
                         alpha=0.2, label='+/- 1 std')

    axes[0].legend()
    axes[0].set_title('Position', weight='bold', fontsize=12)

    # Plot angle distance form goal
    axes[1].set_ylabel('distance from goal', fontsize=11)
    axes[1].grid()

    axes[1].plot(time_steps, mean_a_diff_from_goal, label='mean')
    axes[1].fill_between(time_steps,
                         mean_a_diff_from_goal - std_a_diff_from_goal,
                         mean_a_diff_from_goal + std_a_diff_from_goal,
                         alpha=0.2, label='+/- 1 std')

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
    goal_angle = dataset_states['goal_angle'][0]

    df = dataset_states[['position', 'goal_position', 'step']]
    df[['x_position', 'y_position']] = pd.DataFrame(df['position'].tolist(), index=df.index)

    mean_by_step = df.groupby(['step']).mean()
    std_by_step = df.groupby(['step']).std()

    x_mean = mean_by_step['x_position']
    y_mean = mean_by_step['y_position']
    x_std = std_by_step['x_position']
    y_std = std_by_step['y_position']

    # Plot the evolution of the position over time
    plt.figure()
    plt.xlabel('timestep', fontsize=11)
    plt.ylabel('position', fontsize=11)
    plt.yticks([x_goal_position, y_goal_position])
    plt.grid()

    plt.plot(time_steps, x_mean, label='x mean')
    plt.fill_between(time_steps, x_mean - x_std, x_mean + x_std, alpha=0.2, label='x +/- 1 std')

    plt.plot(time_steps, y_mean, label='y mean')
    plt.fill_between(time_steps, y_mean - y_std, y_mean + y_std, alpha=0.2, label='y +/- 1 std')

    plt.legend()
    plt.title(title, weight='bold', fontsize=14)

    save_visualisation(filename, img_dir)
