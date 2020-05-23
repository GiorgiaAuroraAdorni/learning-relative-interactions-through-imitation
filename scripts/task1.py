import argparse
import os

from dataset import load_dataset, save_dataset, generate_splits
from utils import directory_for_dataset, directory_for_model


def parse_args():
    parser = argparse.ArgumentParser(description='Imitation Learning - Task1')

    parser.add_argument('--gui', action="store_true",
                        help='run simulation using the gui (default: False)')
    parser.add_argument('--n-simulations', type=int, default=1000, metavar='N',
                        help='number of runs for each simulation (default: 1000)')
    parser.add_argument('--initial-poses', default='uniform', choices=['uniform', 'load'],
                        help='choose how to generate the initial positions for each run, '
                             'between uniform, demo and load (default: uniform)')
    parser.add_argument('--initial-poses-file', default='initial_poses.npy',
                        help='name of the file where to store/load the initial poses')
    parser.add_argument('--generate-dataset', action="store_true",
                        help='generate the dataset containing the n_simulations (default: False)')
    parser.add_argument('--goal-object', default="station", choices=['station', 'coloured_station'],
                        help='choose the type of goal object between station and coloured_station (default: station)')
    parser.add_argument('--plots-dataset', action="store_true",
                        help='generate the plots of regarding the dataset (default: False)')
    parser.add_argument('--generate-splits', action="store_true",
                        help='generate dataset splits for training, validation and testing (default: False)')
    # parser.add_argument('--controller', default='all', choices=['all', 'learned', 'omniscient'],
    #                     help='choose the controller for the current execution between all, learned, manual and '
    #                          'omniscient (default: all)')
    parser.add_argument('--controller', default='all', type=str,
                        help='choose the controller for the current execution usually between all, learned and '
                             'omniscient (default: all)')

    parser.add_argument('--dataset-folder', default='datasets', type=str,
                        help='name of the directory containing the datasets (default: datasets)')
    parser.add_argument('--dataset', default='all', type=str,
                        help='choose the datasets to use in the current execution (default: all)')
    parser.add_argument('--models-folder', default='models', type=str,
                        help='name of the directory containing the models (default: models)')
    parser.add_argument('--tensorboard-folder', default='tensorboard', type=str,
                        help='name of the directory containing Tensorboard logs (default: tensorboard)')

    parser.add_argument('--model', default='net1', type=str,
                        help='name of the model (default: net1)')
    parser.add_argument('--arch', default='convnet', choices=['convnet', 'convnet_maxpool'],
                        help='choose the network architecture to use for training the network, '
                             'between convnet and convnet_maxpool (default: convnet)')
    parser.add_argument('--loss', default='mse', choices=['mse', 'smooth_l1'],
                        help='choose the loss function to use for training or evaluating the network, '
                             'between mse and smooth_l1 (default: mse)')
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

    if args.dataset == 'all':
        datasets = [f.path for f in os.scandir(args.dataset_folder) if f.is_dir()]
    else:
        dataset = os.path.join(args.dataset_folder, args.dataset)
        datasets = [dataset]

    for d in datasets:
        for c in controllers:
            run_dir, run_img_dir, run_video_dir = directory_for_dataset(d, c)
            model_dir, model_img_dir, model_video_dir, metrics_path, tboard_dir = directory_for_model(args)

            initial_poses_path = os.path.join(run_dir, args.initial_poses_file)

            if args.initial_poses == 'load':
                from simulations import GenerateSimulationData as sim

                print("Loading initial poses from file '%s'…" % (initial_poses_path))
                initial_poses = sim.load_initial_poses(initial_poses_path)
            else:
                from simulations import GenerateSimulationData as sim

                print("Generating %d initial positions using method '%s'…" % (args.n_simulations, args.initial_poses))
                initial_poses = sim.generate_initial_poses(args.initial_poses, args.n_simulations)
                sim.save_initial_poses(initial_poses_path, initial_poses)

            if args.generate_dataset:
                from simulations import GenerateSimulationData as sim

                print('Generating %s simulations for %s %s controller…' % (args.n_simulations, d, c))
                dataset = sim.generate_simulation(n_simulations=args.n_simulations, controller=c,
                                                  goal_object=args.goal_object, gui=args.gui,
                                                  model_dir=model_dir, initial_poses=initial_poses)

                print('Saving dataset for %s %s controller…' % (d, c))
                save_dataset(run_dir, dataset=dataset)
                print()

            if args.plots_dataset:
                from plots import generate_dataset_plots

                print('Generating plots for %s %s controller…' % (d, c))
                generate_dataset_plots(run_dir, run_img_dir, run_video_dir, args.goal_object)
            if args.generate_splits:
                print('Generating splits…')
                dataset = load_dataset(run_dir)
                splits = generate_splits(dataset)
                save_dataset(run_dir, splits=splits)

            if c == 'omniscient':
                if args.train_net:
                    print('Training model %s…' % args.model)
                    print()

                    from neural_networks import train_net

                    dataset, splits = load_dataset(run_dir, load_splits=True)

                    train_net(dataset, splits, model_dir, metrics_path, tboard_dir,
                              arch=args.arch, loss=args.loss)

                if args.evaluate_net:
                    print('Generating plots for model %s, using loss function "%s"…' % (args.model, args.loss))

                    from simulations import GenerateSimulationData as sim
                    from plots import plot_demo_trajectories
                    n_runs = 7
                    initial_poses = sim.generate_initial_poses('demo', n_runs)
                    generate_args = dict(
                        initial_poses=initial_poses,
                        n_simulations=n_runs,
                        goal_object=args.goal_object,
                        model_dir=model_dir
                    )
                    omniscient_ds = sim.generate_simulation(controller='omniscient', **generate_args)
                    learned_ds = sim.generate_simulation(controller='learned', **generate_args)
                    plot_demo_trajectories(omniscient_ds, learned_ds, args.goal_object, model_video_dir, 'demo-trajectories')

                    from plots import plot_initial_positions
                    plot_initial_positions(args.goal_object, run_dir, model_img_dir, 'initial-positions')

                    from network_evaluation import evaluate_net
                    dataset, splits = load_dataset(run_dir, load_splits=True)
                    evaluate_net(dataset, splits, model_dir, model_img_dir, metrics_path, args.loss)
