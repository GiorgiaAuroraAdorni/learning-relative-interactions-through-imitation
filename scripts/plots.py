import os

import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

import viz
from dataset import load_dataset
from geometry import Point, Transform
from kinematics import to_robot_velocities
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


def plot_distance_from_goal(runs_dir, img_dir, filename):
    """
    :param runs_dir:
    :param img_dir:
    :param filename
    """
    dataset_states = load_dataset(runs_dir)

    time_steps = np.arange(dataset_states.step.max() + 1)

    fig, axes = plt.subplots(nrows=2, figsize=(6.8, 8.4), constrained_layout=True, sharex=True)
    plt.xlabel('timestep', fontsize=11)

    # Plot position distance from goal
    goal_p_dist_by_step = dataset_states.goal_position_distance.groupby('step')
    p_q1, p_q2, p_q3, p_q4, p_median = unpack(goal_p_dist_by_step.quantile([0.25, 0.75, 0.10, 0.90, 0.5]), 'quantile')

    axes[0].set_ylabel('euclidean distance', fontsize=11)
    axes[0].grid()

    ln, = axes[0].plot(time_steps, p_median, label='median')
    axes[0].fill_between(time_steps, p_q1, p_q2, alpha=0.2, label='interquartile range', color=ln.get_color())
    axes[0].fill_between(time_steps, p_q3, p_q4, alpha=0.1, label='interdecile range', color=ln.get_color())

    axes[0].legend()
    axes[0].set_title('Position', weight='bold', fontsize=12)

    # Plot angle distance form goal
    goal_a_dist_by_step = dataset_states.goal_angle_distance.groupby('step')
    a_q1, a_q2, a_q3, a_q4, a_median = unpack(goal_a_dist_by_step.quantile([0.25, 0.75, 0.10, 0.90, 0.5]), 'quantile')

    axes[1].set_ylabel('angle difference', fontsize=11)
    axes[1].grid()

    ln, = axes[1].plot(time_steps, a_median, label='median')
    axes[1].fill_between(time_steps, a_q1, a_q2, alpha=0.2, label='interquartile range', color=ln.get_color())
    axes[1].fill_between(time_steps, a_q3, a_q4, alpha=0.1, label='interdecile range', color=ln.get_color())

    axes[1].legend()
    axes[1].set_title('Angle', weight='bold', fontsize=12)

    save_visualisation(filename, img_dir)


def plot_position_over_time(runs_dir, img_dir, filename):
    """

    :param runs_dir:
    :param img_dir:
    :param filename:
    """
    dataset_states = load_dataset(runs_dir)

    time_steps = np.arange(dataset_states.step.max() + 1)
    x_goal_position, y_goal_position = unpack(dataset_states.goal_position[0], 'axis')
    position_by_step = dataset_states.position.groupby('step').quantile([0.25, 0.75, 0.10, 0.90, 0.5])

    # Plot the evolution of the position over time
    plt.figure(figsize=(7.8, 6.8), constrained_layout=True)

    plt.xlabel('timestep', fontsize=11)
    plt.ylabel('position', fontsize=11)
    plt.yticks([x_goal_position, y_goal_position])
    plt.grid()
    labels = ['median', 'interquartile range', 'interdecile range']

    for quantiles in unpack(position_by_step, "axis"):
        q1, q2, q3, q4, median = unpack(quantiles, "quantile")

        ln, = plt.plot(time_steps, median, label=labels[0])
        plt.fill_between(time_steps, q1, q2, alpha=0.2, label=labels[1],
                       color=ln.get_color())
        plt.fill_between(time_steps, q3, q4, alpha=0.1, label=labels[2],
                         color=ln.get_color())

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    colors = ["w", "w"]
    texts = ["x axis", "y axis"]
    patches = [mpatches.Patch(color=colors[i], label="{:s}".format(texts[i])) for i in range(len(texts))]

    handles = [patches[0], handles[0], handles[2], handles[4], patches[1], handles[1], handles[3], handles[5]]
    labels = [texts[0], labels[0], labels[2], labels[4], texts[1], labels[1], labels[3], labels[5]]

    plt.legend(handles=handles, labels=labels, loc='lower center', fontsize=11, bbox_to_anchor=(0.5, -0.3), ncol=2)

    save_visualisation(filename, img_dir)


def plot_goal_reached_distribution(runs_dir, img_dir, filename):
    """

    :param runs_dir:
    :param img_dir:
    :param filename:
    """
    dataset_states = load_dataset(runs_dir)

    time_steps = np.arange(dataset_states.step.max() + 1)

    states_subset = dataset_states[["step", "goal_reached"]]
    last_steps = states_subset.groupby("run").map(lambda x: x.isel(sample=-1))

    false_label, false_samples = False, last_steps.where(last_steps.goal_reached == False, drop=True)
    true_label, true_samples = True, last_steps.where(last_steps.goal_reached == True, drop=True)

    plt.figure(figsize=(7.8, 4.8), constrained_layout=True)
    plt.hist([true_samples.step, false_samples.step], bins=time_steps, label=[true_label, false_label], stacked=True,
             alpha=0.9)
    plt.ylim(0, plt.ylim()[1] + 1)
    plt.legend()

    plt.xlim(0, dataset_states.step.max() + 1)
    plt.xlabel('timestep', fontsize=11)
    plt.ylabel('samples', fontsize=11)

    save_visualisation(filename, img_dir)


def plot_trajectory(runs_dir, img_dir, filename):
    """

    :param runs_dir:
    :param img_dir:
    :param filename:
    """
    dataset_states = load_dataset(runs_dir)

    run_states = dataset_states.where(dataset_states.run == 0, drop=True)

    goal_position = run_states.goal_position[0]
    goal_angle = run_states.goal_angle[0]

    init_position = run_states.initial_position[0]
    init_angle = run_states.initial_angle[0]

    x_position, y_position = unpack(run_states.position, 'axis')

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)

    ax.set_xlabel('x axis', fontsize=11)
    ax.set_ylabel('y axis', fontsize=11)

    ax.grid()

    obj_points = Point.from_list([
        (-0.5, 1, 1), (1.5, 1, 1), (1.5, 0.5, 1),  (0, 0.5, 1),
        (0, -0.5, 1), (1.5, -0.5, 1), (1.5, -1, 1), (-0.5, -1, 1)
    ])

    obj_tform = Transform.scale(20)
    obj_points = obj_points.transformed(obj_tform).to_euclidean().T

    ax.add_patch(plt.Polygon(obj_points,
                             facecolor=colors.to_rgba([0, 0.5, 0.5], alpha=0.5),
                             edgecolor=[0, 0.5, 0.5],
                             linewidth=1.5,
                             label='docking station'))

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
    ax.set_ylim(-200, 200)
    ax.set_xlim(-300, 300)
    ax.set_aspect('equal')

    plt.legend()
    plt.title('Run 0', fontsize=14, weight='bold')

    save_visualisation(filename, img_dir)


def plot_sensors(runs_dir, video_dir, filename):
    """

    :param runs_dir:
    :param video_dir:
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
        ], suptitle='Run 0')
    ], sources=[marxbot])
    env.show(figsize=(9, 4))

    video_path = os.path.join(video_dir, '%s.mp4' % filename)
    env.save(video_path, dpi=300)


def plot_initial_positions(runs_dir, img_dir, filename):
    """

    :param runs_dir:
    :param img_dir:
    :param filename:
    :return:
    """
    dataset_states = load_dataset(runs_dir)
    step_states = dataset_states.where(dataset_states.step == 0, drop=True)
    x, y = unpack(step_states.initial_position, 'axis')

    plt.figure(figsize=(7.8, 4.8), constrained_layout=True)

    plt.scatter(x, y, alpha=0.2)
    plt.axis('equal')

    plt.xlabel('x axis', fontsize=11)
    plt.ylabel('y axis', fontsize=11)

    save_visualisation(filename, img_dir)


def plot_losses(train_loss, valid_loss, img_dir, filename, scale=None):
    """

    :param train_loss: the training losses
    :param valid_loss: the testing losses
    :param img_dir: directory for the output image
    :param filename:
    :param scale:
    """
    x = np.arange(0, len(train_loss), dtype=int)
    x_ticks = np.arange(0, len(train_loss) + 1, 10, dtype=int)

    plt.figure(figsize=(7.8, 4.8), constrained_layout=True)
    plt.xlabel('epoch', fontsize=11)
    plt.ylabel('loss', fontsize=11)

    plt.xticks(x_ticks)

    plt.plot(x, train_loss, label='train')
    plt.plot(x, valid_loss, label='validation')
    if scale is not None:
        plt.ylim(0, scale)

    plt.yscale('log')
    plt.legend()

    save_visualisation(filename, img_dir)


def plot_target_distribution(y_g, y_p, img_dir, filename):
    """
    
    :param y_g:
    :param y_p:
    :param img_dir:
    :param filename:
    """
    labels = ['groundtruth', 'prediction']

    left_g, right_g = np.split(y_g, 2, axis=1)
    left_p, right_p = np.split(y_p, 2, axis=1)

    fig, axes = plt.subplots(nrows=2, figsize=(6.8, 8.4), constrained_layout=True)

    plt.yscale('log')

    axes[0].set_xlabel('left', fontsize=11)
    left = np.array([left_g, left_p]).reshape(-1, 2)
    axes[0].hist(left, bins=50, label=labels)
    axes[0].legend()
    axes[0].set_title('Left wheel target speed', weight='bold', fontsize=12)

    axes[1].set_xlabel('right', fontsize=11)
    right = np.array([right_g, right_p]).reshape(-1, 2)
    axes[1].hist(right, bins=50, label=labels)
    axes[1].legend()
    axes[1].set_title('Right wheel target speed', weight='bold', fontsize=12)

    save_visualisation(filename, img_dir)


def plot_regressor(y_g, y_p, img_dir, filename):
    """
    :param y_g:
    :param y_p:
    :param img_dir:
    :param filename:
    """
    lin_vel_g, ang_vel_g = to_robot_velocities(y_g[:, 0], y_g[:, 1])
    lin_vel_p, ang_vel_p = to_robot_velocities(y_p[:, 0], y_p[:, 1])

    fig, axes = plt.subplots(nrows=2, figsize=(6.8, 8.4), constrained_layout=True)
    for a in axes:
        a.set_xlabel('groundtruth', fontsize=11)
        a.set_ylabel('prediction', fontsize=11)
        a.axis('equal')

    lr_lin = LinearRegression()
    lr_lin.fit(np.reshape(lin_vel_g, [-1, 1]), np.reshape(lin_vel_p, [-1, 1]))
    score_lin = lr_lin.score(np.reshape(lin_vel_g, [-1, 1]), np.reshape(lin_vel_p, [-1, 1]))

    axes[0].scatter(lin_vel_g, lin_vel_p, alpha=0.3, marker='.', label='sample')
    axes[0].plot(lin_vel_g, lr_lin.predict(np.reshape(lin_vel_g, [-1, 1])), color="orange",
                 label='regression: $R^2=%.3f$' % score_lin)
    axes[0].legend()
    axes[0].set_title('Linear Velocity', weight='bold', fontsize=12)

    lr_ang = LinearRegression()
    lr_ang.fit(np.reshape(ang_vel_g, [-1, 1]), np.reshape(ang_vel_p, [-1, 1]))
    score_ang = lr_ang.score(np.reshape(ang_vel_g, [-1, 1]), np.reshape(ang_vel_p, [-1, 1]))

    axes[1].scatter(ang_vel_g, ang_vel_p, alpha=0.3, marker='.', label='sample')
    axes[1].plot(ang_vel_g, lr_ang.predict(np.reshape(ang_vel_g, [-1, 1])), color="orange",
                 label='regression: $R^2=%.3f$' % score_ang)
    axes[1].legend()
    axes[1].set_title('Angular Velocity', weight='bold', fontsize=12)

    save_visualisation(filename, img_dir)
