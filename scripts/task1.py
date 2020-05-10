import argparse
import os

from dataset import load_dataset, save_dataset, generate_splits
from utils import check_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Imitation Learning - Task1')

    parser.add_argument('--gui', action="store_true",
                        help='run simulation using the gui (default: False)')
    parser.add_argument('--n-simulations', type=int, default=1000, metavar='N',
                        help='number of runs for each simulation (default: 1000)')
    parser.add_argument('--generate-dataset', action="store_true",
                        help='generate the dataset containing the n_simulations (default: False)')
    parser.add_argument('--plots-dataset', action="store_true",
                        help='generate the plots of regarding the dataset (default: False)')
    parser.add_argument('--generate-splits', action="store_true",
                        help='generate dataset splits for training, validation and testing (default: False)')
    parser.add_argument('--controller', default='all', choices=['all', 'learned', 'omniscient'],
                        help='choose the controller for the current execution between all, learned, manual and '
                             'omniscient (default: all)')

    parser.add_argument('--dataset-folder', default='datasets', type=str,
                        help='name of the directory containing the datasets (default: datasets)')
    parser.add_argument('--model-folder', default='models', type=str,
                        help='name of the directory containing the models (default: models)')

    parser.add_argument('--model', default='net1', type=str,
                        help='name of the model (default: net1)')
    parser.add_argument('--train-net', action="store_true",
                        help='train the model  (default: False)')
    parser.add_argument('--evaluate-net', action="store_true",
                        help='evaluate the model  (default: False)')
    parser.add_argument('--plots-net', action="store_true",
                        help='generate the plots of regarding the model (default: False)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    runs_dir = os.path.join(args.dataset_folder)
    check_dir(runs_dir)

    omniscient_controller = "omniscient-controller"

    runs_dir_omniscient = os.path.join(runs_dir, omniscient_controller)
    check_dir(runs_dir_omniscient)

    img_dir_omniscient = os.path.join(runs_dir_omniscient, 'images')
    check_dir(img_dir_omniscient)

    video_dir_omniscient = os.path.join(runs_dir_omniscient, 'videos')
    check_dir(video_dir_omniscient)

    model_dir = os.path.join(args.models_folder, args.model)
    check_dir(model_dir)

    img_dir_model = os.path.join(model_dir, 'images')
    check_dir(img_dir_model)

    metrics_path = os.path.join(model_dir, 'metrics.pkl')

    if args.controller == 'all' or args.controller == 'omniscient':
        if args.generate_dataset:
            from simulations import GenerateSimulationData as sim

            print('Generating n_simulations for %s…' % omniscient_controller)
            dataset = sim.generate_simulation(
                n_simulations=args.n_simulations, controller=omniscient_controller, args=args
            )
            print('Saving dataset for %s…' % omniscient_controller)
            save_dataset(runs_dir_omniscient, dataset=dataset)
            print()

        if args.plots_dataset:
            from plots import plot_distance_from_goal, plot_position_over_time, \
    plot_goal_reached_distribution, plot_sensors, plot_trajectory, plot_initial_positions, plot_trajectories

            print('Generating plots for %s…' % omniscient_controller)

            # plot_distance_from_goal(runs_dir_omniscient, img_dir_omniscient,
            #                         'distances-from-goal-%s' % omniscient_controller)
            #
            # plot_position_over_time(runs_dir_omniscient, img_dir_omniscient,
            #                         'pose-over-time-%s' % omniscient_controller)
            #
            # plot_goal_reached_distribution(runs_dir_omniscient, img_dir_omniscient,
            #                                'goal-reached-%s' % omniscient_controller)
            #
            # plot_trajectory(runs_dir_omniscient, img_dir_omniscient, 'robot-trajectory-%s' % omniscient_controller)

            plot_trajectories(runs_dir_omniscient, img_dir_omniscient,
                              '10-robot-trajectories-%s' % omniscient_controller)

            # plot_sensors(runs_dir_omniscient, video_dir_omniscient,
            #              'sensors-control-response-over-time-%s' % omniscient_controller)

            # plot_initial_positions(runs_dir_omniscient, img_dir_omniscient, 'initial-positions')

        if args.generate_splits:
            print('Generating splits…')
            dataset = load_dataset(runs_dir_omniscient)
            splits = generate_splits(dataset)
            save_dataset(runs_dir_omniscient, splits=splits)

        if args.train_net:
            print('Training model %s…' % args.model)
            from neural_networks import train_net

            dataset, splits = load_dataset(runs_dir_omniscient, load_splits=True)

            train_net(dataset, splits, model_dir, metrics_path)

        if args.evaluate_net:
            print('Generating plots for model %s…' % args.model)
            from network_evaluation import evaluate_net

            dataset, splits = load_dataset(runs_dir_omniscient, load_splits=True)
            evaluate_net(dataset, splits, model_dir, img_dir_model, metrics_path)

