import os
import itertools

import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import viz
from dataset import load_dataset
from geometry import Point, Transform
from kinematics import to_robot_velocities
from utils import unpack


def generate_dataset_plots(run_dir, img_dir, video_dir, goal_object):
    """

    :param run_dir:
    :param img_dir:
    :param video_dir:
    :param goal_object
    """
    plot_distance_from_goal(run_dir, img_dir, 'distances-from-goal')
    plot_position_over_time(run_dir, img_dir, 'pose-over-time')
    plot_goal_reached_distribution(run_dir, img_dir, 'goal-reached')
    plot_trajectory(goal_object, run_dir, img_dir, 'robot-trajectory')
    plot_trajectories(goal_object, run_dir, img_dir, '10-robot-trajectories')
    plot_positions_scatter(goal_object, run_dir, img_dir, 'initial-final-positions')
    plot_positions_heatmap(goal_object, run_dir, img_dir, 'positions-heatmap')
    for r in range(10):
        plot_sensors(goal_object, run_dir, video_dir, 'sensors-control-response-over-time', run_id=r)


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

    fig, axes = plt.subplots(ncols=2, figsize=(10.8, 4.8), constrained_layout=True)

    # Plot position distance from goal
    goal_p_dist_by_step = dataset_states.goal_position_distance.groupby('step')
    p_q1, p_q2, p_q3, p_q4, p_median = unpack(goal_p_dist_by_step.quantile([0.25, 0.75, 0.10, 0.90, 0.5]), 'quantile')

    axes[0].set_ylabel('euclidean distance', fontsize=11)
    axes[0].set_xlabel('timestep', fontsize=11)
    axes[0].grid()

    ln, = axes[0].plot(time_steps, p_median, label='median')
    axes[0].fill_between(time_steps, p_q1, p_q2, alpha=0.2, label='interquartile range', color=ln.get_color())
    axes[0].fill_between(time_steps, p_q3, p_q4, alpha=0.1, label='interdecile range', color=ln.get_color())
    axes[0].set_ylim(0)

    axes[0].legend()
    axes[0].set_title('Position', weight='bold', fontsize=12)

    # Plot angle distance form goal
    goal_a_dist_by_step = dataset_states.goal_angle_distance.groupby('step')
    a_q1, a_q2, a_q3, a_q4, a_median = unpack(goal_a_dist_by_step.quantile([0.25, 0.75, 0.10, 0.90, 0.5]), 'quantile')

    axes[1].set_ylabel('angle difference', fontsize=11)
    axes[1].set_xlabel('timestep', fontsize=11)
    axes[1].grid()
    axes[1].set_ylim(0, 3)

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

    handles = [patches[0], handles[0], handles[2], handles[3], patches[1], handles[1], handles[4], handles[5]]
    labels = [texts[0], labels[0], labels[2], labels[3], texts[1], labels[1], labels[4], labels[5]]

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
    last_steps = states_subset.groupby("run").map(lambda x: x.isel(sample=[-1]))

    false_label, false_samples = False, last_steps.where(last_steps.goal_reached == False, drop=True)
    true_label, true_samples = True, last_steps.where(last_steps.goal_reached == True, drop=True)

    plt.figure(figsize=(7.8, 4.8), constrained_layout=True)
    plt.hist([true_samples.step, false_samples.step], bins=time_steps, label=[true_label, false_label], stacked=True,
             alpha=0.9)
    plt.ylim(0, plt.ylim()[1] + 1)
    plt.legend()

    plt.xlim(0, dataset_states.step.max() + 1)
    plt.xlabel('timestep', fontsize=11)
    plt.ylabel('runs', fontsize=11)

    save_visualisation(filename, img_dir)


def draw_docking_station(ax, goal_object='station'):
    """

    :param ax:
    :param goal_object
    """
    obj_tform = Transform.scale(20) @ Transform.translate(-0.85, 0)

    if goal_object == 'station':
        obj_points = Point.from_list([
            (0, 1, 1), (2, 1, 1), (2, 0.5, 1), (0.5, 0.5, 1),
            (0.5, -0.5, 1), (2, -0.5, 1), (2, -1, 1), (0, -1, 1)
        ])

        obj_points = obj_points.transformed(obj_tform).to_euclidean().T

        ax.add_patch(plt.Polygon(
            obj_points,
            facecolor=colors.to_rgba([0, 0.5, 0.5], alpha=0.5),
            edgecolor=[0, 0.5, 0.5],
            linewidth=1.5
        ))
    elif goal_object == 'coloured_station':
        face_colours = [[0.839, 0.153, 0.157], [1.000, 0.895, 0.201], [0.173, 0.627, 0.173]]

        obj_points = Point.from_list([(0, 1, 1), (0, 0.5, 1), (2, 0.5, 1), (2, 1, 1),
                                      (0, -1, 1), (0, -0.5, 1), (2, -0.5, 1), (2, -1, 1),
                                      (0, 0.5, 1), (0, -0.5, 1), (0.5, -0.5, 1), (0.5, 0.5, 1)])

        obj_points = obj_points.transformed(obj_tform).to_euclidean().T

        arm1_points = obj_points[:4]
        arm2_points = obj_points[4:8]
        back_points = obj_points[8:]

        ax.add_patch(plt.Polygon(
            back_points,
            facecolor=colors.to_rgba(face_colours[1], alpha=0.5),
            edgecolor=face_colours[1],
            linewidth=1.5
        ))
        ax.add_patch(plt.Polygon(
            arm1_points,
            facecolor=colors.to_rgba(face_colours[0], alpha=0.5),
            edgecolor=face_colours[0],
            linewidth=1.5
        ))
        ax.add_patch(plt.Polygon(
            arm2_points,
            facecolor=colors.to_rgba(face_colours[2], alpha=0.5),
            edgecolor=face_colours[2],
            linewidth=1.5
        ))

    else:
        raise ValueError("Invalid value for goal_object")


def draw_marxbot(ax, position=None, angle=None, *, label=None, colour=None, radius=8.5, show_range=False, range_radius=150.0):
    """

    :param ax:
    :param position:
    :param angle:
    :param label:
    :param colour:
    :param radius:
    :param show_range:
    :param range_radius:
    :return:
    """
    if colour is None:
        if label == 'goal position':
            colour = 'tab:orange'
        else:
            colour = 'tab:blue'

    class MarxbotPatch:
        _points = Point.from_list([
            Point.ORIGIN,
            [1, -3, 1], [6, 0, 1], [1, 3, 1]
        ])

        def __init__(self, ax):
            points = self._points.to_euclidean().T

            self.circle = plt.Circle(
                points[0], radius,
                facecolor=colors.to_rgba(colour, alpha=0.5),
                edgecolor=colour, linewidth=1.5,
                label=label
            )
            self.arrow = plt.Polygon(
                points[1:],
                facecolor='none',
                edgecolor=colour
            )

            ax.add_patch(self.circle)
            ax.add_patch(self.arrow)

            if show_range:
                self.range_circle = plt.Circle(
                    points[0], 150.0,
                    facecolor=colors.to_rgba(colour, alpha=0.1),
                    edgecolor=colour, linewidth=0.5,
                    label='sensors range'
                )

                ax.add_patch(self.range_circle)

        def update(self, position, angle):
            tform = Transform.pose_transform(position, angle)
            points = self._points.transformed(tform).to_euclidean().T

            self.circle.center = points[0]
            self.arrow.xy = points[1:]

            if show_range:
                self.range_circle.center = points[0]

    patch = MarxbotPatch(ax)

    if position is not None and angle is not None:
        patch.update(position, angle)

    return patch


def plot_trajectory(goal_object, runs_dir, img_dir, filename, run_id=0):
    """

    :param goal_object
    :param runs_dir:
    :param img_dir:
    :param filename:
    :param run_id:
    """
    dataset_states = load_dataset(runs_dir)
    dataset_states = dataset_states[['goal_position', 'goal_angle', 'position', 'initial_position', 'initial_angle']]

    run_states = dataset_states.where(dataset_states.run == run_id, drop=True)

    init_position = run_states.initial_position[run_id]
    init_angle = run_states.initial_angle[run_id]

    goal_position = run_states.goal_position[run_id]
    goal_angle = run_states.goal_angle[run_id]

    x_position, y_position = unpack(run_states.position, 'axis')

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    ax.set_xlabel('x axis', fontsize=11)
    ax.set_ylabel('y axis', fontsize=11)
    ax.grid()

    draw_docking_station(ax, goal_object)
    draw_marxbot(ax, init_position, init_angle, label='initial position')
    draw_marxbot(ax, goal_position, goal_angle, label='goal position')
    plt.plot(x_position, y_position, color='black', label='trajectory', linewidth=1)

    ax.set_ylim(-220, 220)
    ax.set_xlim(-250, 250)
    ax.set_aspect('equal')

    plt.legend()
    plt.title('Run %d' % run_id, fontsize=14, weight='bold')

    save_visualisation(filename, img_dir)


def plot_trajectories(goal_object, runs_dir, img_dir, filename, n_runs=10):
    """

    :param runs_dir:
    :param img_dir:
    :param filename:
    :param goal_object
    :param n_runs:
    """
    dataset_states = load_dataset(runs_dir)
    dataset_states = dataset_states[['goal_position', 'goal_angle', 'position', 'initial_position', 'initial_angle']]

    runs = np.arange(n_runs)
    run_states = dataset_states.where(dataset_states.run.isin(runs), drop=True)

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)

    ax.set_xlabel('x axis', fontsize=11)
    ax.set_ylabel('y axis', fontsize=11)

    ax.grid()

    for run_id, run in run_states.groupby('run'):
        init_position = run.initial_position[run_id]
        init_angle = run.initial_angle[run_id]

        x_position, y_position = unpack(run.position, 'axis')

        if run_id == 0:
            goal_position = run_states.goal_position[run_id]
            goal_angle = run_states.goal_angle[run_id]

            draw_docking_station(ax, goal_object)
            draw_marxbot(ax, goal_position, goal_angle, label='goal position')

            draw_marxbot(ax, init_position, init_angle, label='initial positions')
            plt.plot(x_position, y_position, color='black', label='trajectories', linewidth=1)
        else:
            draw_marxbot(ax, init_position, init_angle)
            plt.plot(x_position, y_position, color='black', linewidth=1)

    ax.set_ylim(-220, 220)
    ax.set_xlim(-250, 250)
    ax.set_aspect('equal')

    plt.legend()
    save_visualisation(filename, img_dir)


def plot_sensors(goal_object, runs_dir, video_dir, filename, run_id=0):
    """

    :param goal_object:
    :param runs_dir:
    :param video_dir:
    :param filename:
    :param run_id
    """
    dataset_states = load_dataset(runs_dir)
    run_states = dataset_states.where(dataset_states.run == run_id, drop=True)

    marxbot = viz.DatasetSource(run_states)

    # Create the visualizations
    env = viz.FuncAnimationEnv([
        viz.GridLayout((1, 3), [
            viz.TrajectoryViz(marxbot, goal_object=goal_object),
            viz.LaserScannerViz(marxbot),
            viz.ControlSignalsViz(marxbot)
        ], suptitle='Run %d' % run_id)
    ], sources=[marxbot])
    env.show(figsize=(14, 4))

    video_path = os.path.join(video_dir, '%s-%d.mp4' % (filename, run_id))
    env.save(video_path, dpi=300)


def plot_demo_trajectories(omniscient_ds, learned_ds, goal_object, video_dir, filename):
    omniscient_bots = [viz.DatasetSource(run) for (run_id, run) in omniscient_ds.groupby('run')]
    learned_bots = [viz.DatasetSource(run) for (run_id, run) in learned_ds.groupby('run')]
    trajectory_args = dict(
        goal_object=goal_object,
        show_goals='first',
    )

    # Create the visualizations
    env = viz.FuncAnimationEnv([
        viz.GridLayout((1, 2), [
            viz.TrajectoryViz(*omniscient_bots, colours='tab:blue', title="Omniscient Trajectories", **trajectory_args),
            viz.TrajectoryViz(*learned_bots, colours='tab:purple', title="Learned Trajectories", **trajectory_args),
        ], sharey=True)
    ], sources=omniscient_bots + learned_bots)
    env.show(figsize=(9.8, 4))

    video_path = os.path.join(video_dir, '%s.mp4' % filename)
    env.save(video_path, dpi=300)


def plot_positions_scatter(goal_object, runs_dir, img_dir, filename):
    """

    :param goal_object
    :param runs_dir:
    :param img_dir:
    :param filename:
    """
    dataset_states = load_dataset(runs_dir)
    dataset_states = dataset_states[['goal_position', 'goal_angle', 'position', 'initial_position', 'initial_angle']]

    fig, axes = plt.subplots(ncols=2, figsize=(9.8, 4.8), constrained_layout=True, sharey='row')
    plt.ylim(-220, 220)

    # initial positions
    step_states = dataset_states.where(dataset_states.step == 0, drop=True)
    x, y = unpack(step_states.initial_position, 'axis')
    label = 'initial positions'

    goal_position = step_states.goal_position[0]
    goal_angle = step_states.goal_angle[0]

    radius = 8.5
    axes[0].scatter(x, y, alpha=0.1, label=label, marker='o', s=(radius * np.pi) ** 2 / 5,
                    facecolor=colors.to_rgba('tab:blue', alpha=0.1), edgecolor='none')
    axes[0].set_ylabel('y axis', fontsize=11)
    draw_docking_station(axes[0], goal_object)
    draw_marxbot(axes[0], goal_position, goal_angle, label='goal position')

    axes[0].set_xlim(-250, 250)
    axes[0].set_aspect('equal')

    axes[0].set_xlabel('x axis', fontsize=11)
    axes[0].legend()

    # final positions
    step_states = dataset_states.groupby("run").map(lambda x: x.isel(sample=[-1]))
    x, y = unpack(step_states.position, 'axis')

    goal_position = step_states.goal_position[0]
    goal_angle = step_states.goal_angle[0]

    axes[1].scatter(x, y, label='final positions', marker='o', s=(radius * np.pi) ** 2 / 5,
                    facecolor=colors.to_rgba('tab:blue', alpha=0.1), edgecolor='none')
    draw_docking_station(axes[1], goal_object)
    draw_marxbot(axes[1], goal_position, goal_angle, label='goal position')

    axes[1].set_xlim(-250, 250)
    axes[1].set_aspect('equal')

    axes[1].set_xlabel('x axis', fontsize=11)
    axes[1].legend()

    save_visualisation(filename, img_dir)


def plot_initial_positions(goal_object, runs_dir, img_dir, filename):
    """
    :param goal_object
    :param runs_dir:
    :param img_dir:
    :param filename:
    """
    dataset_states, splits = load_dataset(runs_dir, load_splits=True)
    step_states = dataset_states.where(dataset_states.step == 0, drop=True)

    plt.figure(figsize=(7.8, 4.8), constrained_layout=True)

    radius = 8.5
    for i, name in enumerate(splits.split_names):
        split_states = step_states.where(splits == i)
        x, y = unpack(split_states.initial_position, 'axis')
        plt.plot(x, y, 'o', label=name, alpha=0.1, markersize=(radius * np.pi) / 2, markeredgecolor='none')

    ax = plt.gca()

    draw_docking_station(ax, goal_object)

    goal_position = step_states.goal_position[0]
    goal_angle = step_states.goal_angle[0]

    draw_marxbot(ax, goal_position, goal_angle, label='goal position')

    ax.set_ylim(-220, 220)
    ax.set_xlim(-250, 250)
    ax.set_aspect('equal')
    plt.legend()

    plt.xlabel('x axis', fontsize=11)
    plt.ylabel('y axis', fontsize=11)

    save_visualisation(filename, img_dir)


def plot_goal_positions(goal_object, runs_dir, img_dir, filename):
    """
    :param goal_object
    :param runs_dir:
    :param img_dir:
    :param filename:
    """
    dataset_states, splits = load_dataset(runs_dir, load_splits=True)
    step_states = dataset_states.where(dataset_states.step == 0, drop=True)

    plt.figure(figsize=(7.8, 4.8), constrained_layout=True)

    radius = 8.5
    for i, name in enumerate(splits.split_names):
        split_states = step_states.where(splits == i)
        x, y = unpack(split_states.goal_position, 'axis')
        plt.plot(x, y, 'o', label=name, alpha=0.1, markersize=(radius * np.pi) / 2, markeredgecolor='none')

    ax = plt.gca()

    draw_docking_station(ax, goal_object)

    ax.set_ylim(-220, 220)
    ax.set_xlim(-250, 250)
    ax.set_aspect('equal')
    plt.legend()

    plt.xlabel('x axis', fontsize=11)
    plt.ylabel('y axis', fontsize=11)

    save_visualisation(filename, img_dir)


def plot_positions_heatmap(goal_object, runs_dir, img_dir, filename):
    """

    :param goal_object:
    :param runs_dir:
    :param img_dir:
    :param filename:
    """

    dataset_states = load_dataset(runs_dir)

    x, y = unpack(dataset_states.position, 'axis')

    n_bins = 100
    grid_x, bins_x = pd.cut(x.data, n_bins, retbins=True)
    grid_y, bins_y = pd.cut(y.data, n_bins, retbins=True)

    grid = np.stack([grid_y.codes, grid_x.codes])
    unique, counts = np.unique(grid, axis=-1, return_counts=True)

    mesh = np.zeros([n_bins, n_bins])
    mesh[unique[0], unique[1]] = counts

    plt.figure()

    cmap = plt.get_cmap('viridis')
    cmap.set_over('w')

    plt.pcolormesh(bins_x, bins_y, mesh, cmap=cmap, norm=colors.PowerNorm(0.5), vmax=200)

    cbar = plt.colorbar()
    cbar.set_label('samples per grid cell (clipped)', labelpad=15)

    plt.axis('image')
    plt.xlabel('x axis', fontsize=11)
    plt.ylabel('y axis', fontsize=11)

    draw_docking_station(plt.gca(), goal_object)

    save_visualisation(filename, img_dir)


def plot_losses(train_loss, valid_loss, img_dir, filename):
    """

    :param train_loss: the training losses
    :param valid_loss: the testing losses
    :param img_dir: directory for the output image
    :param filename:
    """
    x = np.arange(0, len(train_loss), dtype=int)

    plt.figure(figsize=(7.8, 4.8), constrained_layout=True)
    plt.xlabel('epoch', fontsize=11)
    plt.ylabel('loss', fontsize=11)

    plt.plot(x, train_loss, label='train')
    plt.plot(x, valid_loss, label='validation')
    plt.ylim(0, max(min(train_loss), min(valid_loss)) * 10)
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

    fig, axes = plt.subplots(ncols=2, figsize=(10.8, 4.8), constrained_layout=True)

    plt.yscale('log')

    left = np.array([left_g, left_p]).reshape(-1, 2)
    axes[0].hist(left, bins=50, label=labels)
    axes[0].legend()
    axes[0].set_title('Left wheel target speed', weight='bold', fontsize=12)

    right = np.array([right_g, right_p]).reshape(-1, 2)
    axes[1].hist(right, bins=50, label=labels)
    axes[1].legend()
    axes[1].set_title('Right wheel target speed', weight='bold', fontsize=12)

    save_visualisation(filename, img_dir)


def plot_wheel_speed_over_time(y_g, y_p, img_dir, filename):
    """
    :param y_g
    :param y_p
    :param img_dir:
    :param filename
    """
    labels = ['groundtruth', 'prediction']

    # extract 1000 subsamples
    left_g, right_g = np.split(y_g[:1000, ], 2, axis=1)
    left_p, right_p = np.split(y_p[:1000, ], 2, axis=1)

    fig, axes = plt.subplots(ncols=2, figsize=(10.8, 4.8), constrained_layout=True)

    left = np.array([left_g, left_p]).reshape(-1, 2)
    right = np.array([right_g, right_p]).reshape(-1, 2)

    for idx, label in enumerate(labels):
        axes[0].plot(left[:, idx], label=label)
        axes[1].plot(right[:, idx], label=label)

    y_min = min(left.min(), right.min())
    y_max = max(left.max(), right.max())

    axes[0].set_ylim(y_min - 10, y_max + 10)
    axes[0].legend()
    axes[0].set_title('Left wheel target speed over time', weight='bold', fontsize=12)
    axes[0].set_xlabel('sample')
    axes[0].set_ylabel('left wheel speed [cm/s]')

    axes[1].set_ylim(y_min - 10, y_max + 10)
    axes[1].legend()
    axes[1].set_title('Right wheel target speed over time', weight='bold', fontsize=12)
    axes[1].set_xlabel('sample')
    axes[1].set_ylabel('right wheel speed [cm/s]')

    save_visualisation(filename, img_dir)


def plot_velocities_over_time(y_g, y_p, img_dir, filename):
    """

    :param y_g:
    :param y_p:
    :param img_dir:
    :param filename:
    :return:
    """
    labels = ['groundtruth', 'prediction']

    # extract 1000 subsamples
    lin_vel_g, ang_vel_g = to_robot_velocities(y_g[:1000, 0], y_g[:1000, 1])
    lin_vel_p, ang_vel_p = to_robot_velocities(y_p[:1000, 0], y_p[:1000, 1])

    fig, axes = plt.subplots(ncols=2, figsize=(10.8, 4.8), constrained_layout=True)

    lin = np.array([lin_vel_g, lin_vel_p]).reshape(-1, 2)
    ang = np.array([ang_vel_g, ang_vel_p]).reshape(-1, 2)

    for idx, label in enumerate(labels):
        axes[0].plot(lin[:, idx], label=label)
        axes[1].plot(ang[:, idx], label=label)

    axes[0].set_ylim(lin.min() - 10, lin.max() + 10)
    axes[0].legend()
    axes[0].set_title('Linear velocity over time', weight='bold', fontsize=12)
    axes[0].set_xlabel('sample')
    axes[0].set_ylabel('linear velocity [cm/s]')

    axes[1].set_ylim(ang.min() - 1, ang.max() + 1)
    axes[1].legend()
    axes[1].set_title('Angular velocity over time', weight='bold', fontsize=12)
    axes[1].set_xlabel('sample')
    axes[1].set_ylabel('angular velocity [rad/s]')

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

    fig, axes = plt.subplots(ncols=2, figsize=(10.8, 4.8), constrained_layout=True)
    for a in axes:
        a.set_xlabel('groundtruth', fontsize=11)
        a.set_ylabel('prediction', fontsize=11)
        a.set_aspect('equal', adjustable='datalim')

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


def plot_losses_distribution(train_losses, valid_losses, img_dir, filename):
    """

    :param valid_losses:
    :param train_losses:
    :param img_dir:
    :param filename:
    """
    plt.figure(figsize=(7.8, 4.8), constrained_layout=True)
    plt.hist([train_losses, valid_losses], label=['train', 'validation'], alpha=0.9)
    plt.yscale('log')
    plt.ylim(0.1, plt.ylim()[1] + 1)
    plt.legend()

    plt.xlabel('loss value', fontsize=11)
    plt.ylabel('samples', fontsize=11)

    save_visualisation(filename, img_dir)
