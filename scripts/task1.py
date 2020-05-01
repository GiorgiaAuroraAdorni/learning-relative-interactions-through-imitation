import argparse
import os

from plots import plot_distance_from_goal, plot_position_over_time
from utils import check_dir
from simulations import GenerateSimulationData as sim


def Parse():
    """

    :return args
    """
    parser = argparse.ArgumentParser(description='PyTorch FrontNet')

    parser.add_argument('--gui', action="store_true",
                        help='run simulation using the gui (default: False)')
    parser.add_argument('--n-simulations', type=int, default=1000, metavar='N',
                        help='number of runs for each simulation (default: 1000)')
    parser.add_argument('--generate-dataset', action="store_true",
                        help='generate the dataset containing the n_simulations (default: False)')
    parser.add_argument('--plots-dataset', action="store_true",
                        help='generate the plots of regarding the dataset (default: False)')
    parser.add_argument('--controller', default='all', choices=['all', 'learned', 'omniscient'],
                        help='choose the controller for the current execution between all, learned, manual and '
                             'omniscient (default: all)')
    parser.add_argument('--dataset-folder', default='datasets/', type=str,
                        help='name of the directory containing the datasets (default: datasets/)')

    parser.add_argument('--train-net', action="store_true",
                        help='train the model  (default: False)')
    parser.add_argument('--model', default='net1', type=str,
                        help='name of the model (default: net1)')
    parser.add_argument('--generate-split', action="store_true",
                        help='generate the indices for the split of the dataset (default: False)')
    parser.add_argument('--plots-net', action="store_true",
                        help='generate the plots of regarding the model (default: False)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = Parse()

    runs_dir = os.path.join(args.dataset_folder)
    check_dir(runs_dir)

    omniscient_controller = "omniscient-controller"

    runs_dir_omniscient = os.path.join(runs_dir, omniscient_controller)
    check_dir(runs_dir_omniscient)

    img_dir_omniscient = os.path.join(runs_dir_omniscient, 'images')
    check_dir(img_dir_omniscient)

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
                                    'pose-over-time-s%s' % omniscient_controller)

