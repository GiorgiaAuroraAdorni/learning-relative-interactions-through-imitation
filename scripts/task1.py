import argparse
import os

from plots import plot_distance_from_goal, plot_position_over_time, plot_goal_reached_distribution, plot_sensors
from utils import check_dir
from simulations import GenerateSimulationData as sim


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
    parser.add_argument('--dataset-folder', default='datasets/', type=str,
                        help='name of the directory containing the datasets (default: datasets/)')

    parser.add_argument('--train-net', action="store_true",
                        help='train the model  (default: False)')
    parser.add_argument('--model', default='net1', type=str,
                        help='name of the model (default: net1)')
    parser.add_argument('--plots-net', action="store_true",
                        help='generate the plots of regarding the model (default: False)')

    args = parser.parse_args()

    return args


def generate_splits(dataset_path, coord='run', splits=None):
    import numpy as np
    import xarray as xr

    if splits is None:
        splits = {
            "train": 0.7,
            "validation": 0.15,
            "test": 0.15
        }

    names = list(splits.keys())
    codes = np.arange(len(splits), dtype=np.int8)
    probs = list(splits.values())

    dataset = xr.open_dataset(dataset_path)

    unique, unique_inverse = np.unique(dataset[coord], return_inverse=True)
    n_indices = unique.size

    unique_assigns = np.random.choice(codes, n_indices, p=probs)
    assigns = unique_assigns[unique_inverse]

    coords = dataset[coord].coords
    attrs = {
        "split_names": names
    }

    splits = xr.DataArray(assigns, name="split", coords=coords, attrs=attrs)

    splits_path = os.path.splitext(dataset_path)[0] + '.splits.nc'
    splits.to_netcdf(splits_path)


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

    if args.controller == 'all' or args.controller == 'omniscient':
        if args.generate_dataset:
            print('Generating n_simulations for %s…' % omniscient_controller)
            sim.generate_simulation(runs_dir_omniscient, n_simulations=args.n_simulations, controller=omniscient_controller,
                                    args=args)

        if args.plots_dataset:
            print('\nGenerating plots for %s…' % omniscient_controller)

            plot_distance_from_goal(runs_dir_omniscient, img_dir_omniscient,
                                    'Robot distance from goal - %s' % omniscient_controller,
                                    'distances-from-goal-%s' % omniscient_controller)

            plot_position_over_time(runs_dir_omniscient, img_dir_omniscient,
                                    'Robot position over time - %s' % omniscient_controller,
                                    'pose-over-time-%s' % omniscient_controller)

            plot_goal_reached_distribution(runs_dir_omniscient, img_dir_omniscient,
                                           'Distribution of the goal reached - %s' % omniscient_controller,
                                           'goal-reached-%s' % omniscient_controller)

            plot_sensors(runs_dir_omniscient, video_dir_omniscient,
                         'Laser scanner response over time - %s' % omniscient_controller,
                         'laser-scanner-response-over-time-%s' % omniscient_controller)

        if args.generate_splits:
            dataset_path = os.path.join(runs_dir_omniscient, 'simulation.nc')
            generate_splits(dataset_path)
