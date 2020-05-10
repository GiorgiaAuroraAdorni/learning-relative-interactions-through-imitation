import argparse

from dataset import load_dataset, save_dataset, generate_splits
from utils import directory_for_dataset, directory_for_model


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
                        help='generate the plots regarding the model  (default: False)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if args.controller == 'all':
        controllers = ['omniscient', 'learned']
    else:
        controllers = [args.controller]

    for controller in controllers:
        run_dir, run_img_dir, run_video_dir = directory_for_dataset(args, controller)
        model_dir, model_img_dir, model_video_dir, metrics_path = directory_for_model(args)

        if args.generate_dataset:
            from simulations import GenerateSimulationData as sim

            print('Generating %s simulations for %s controller…' % (args.n_simulations, controller))
            dataset = sim.generate_simulation(n_simulations=args.n_simulations,
                                              controller=controller, args=args,
                                              model_dir=model_dir)

            print('Saving dataset for %s controller…' % controller)
            save_dataset(run_dir, dataset=dataset)
            print()

        if args.plots_dataset:
            from plots import generate_dataset_plots

            print('Generating plots for %s controller…' % controller)
            generate_dataset_plots(run_dir, run_img_dir, run_video_dir)

        if args.generate_splits:
            print('Generating splits…')
            dataset = load_dataset(run_dir)
            splits = generate_splits(dataset)
            save_dataset(run_dir, splits=splits)

        if controller == 'omniscient':
            if args.train_net:
                print('Training model %s…' % args.model)
                from neural_networks import train_net

                dataset, splits = load_dataset(run_dir, load_splits=True)

                train_net(dataset, splits, model_dir, metrics_path)

            if args.evaluate_net:
                print('Generating plots for model %s…' % args.model)
                from network_evaluation import evaluate_net

                dataset, splits = load_dataset(run_dir, load_splits=True)
                evaluate_net(dataset, splits, model_dir, model_img_dir, metrics_path)
