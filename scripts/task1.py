import argparse
import os

from utils import check_dir
from simulations import GenerateSimulationData as sim


def Parse():
    """

    :return args
    """
    parser = argparse.ArgumentParser(description='PyTorch FrontNet')

    parser.add_argument('--gui', action="store_true",
                        help='run simulation using the gui (default: False)')
    parser.add_argument('--simulations', type=int, default=1000, metavar='N',
                        help='number of runs for each simulation (default: 1000)')
    parser.add_argument('--generate-dataset', action="store_true",
                        help='generate the dataset containing the simulations (default: False)')
    parser.add_argument('--plots-dataset', action="store_true",
                        help='generate the plots of regarding the dataset (default: False)')
    parser.add_argument('--check-dataset', action="store_true",
                        help='generate the plots that check the dataset conformity (default: False)')

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

    img_dir = os.path.join(runs_dir, 'images')
    check_dir(img_dir)

    omniscient_controller = "omniscient-controller"

    runs_dir_omniscient = os.path.join(runs_dir, omniscient_controller)
    check_dir(runs_dir_omniscient)

    img_dir_omniscient = os.path.join(runs_dir_omniscient, 'images')
    check_dir(img_dir_omniscient)

    if args.controller == 'all' or args.controller == 'omniscient':
        if args.generate_dataset:
            print('Generating simulations for %s…' % omniscient_controller)
            sim.generate_simulation(simulations=args.simulations, controller=omniscient_controller, args=args)
