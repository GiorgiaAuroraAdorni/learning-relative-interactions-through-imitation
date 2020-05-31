# learning-relative-interactions-through-imitation

> Project for Robotics course @ USI 19/20.

#### Contributors

**Giorgia Adorni** - giorgia.adorni@usi.ch  [GiorgiaAuroraAdorni](https://github.com/GiorgiaAuroraAdorni)

**Elia Cereda** - elia.cereda@usi.ch  [EliaCereda](https://github.com/EliaCereda)

#### Prerequisites

- Python 3
- Enki
- PyTorch

#### Installation

To install Enki follow the following instructions: https://jeguzzi.github.io/enki/intro.html.

Clone our repository and install the requirements

```sh
$ git clone https://github.com/GiorgiaAuroraAdorni/learning-relative-interactions-through-imitation
$ cd learning-relative-interactions-through-imitation
$ pip install -r requirements.txt
```

#### Usage

To receive help on how to run the script, execute:

```sh
$ python scripts/task1.py --help

> usage: task1.py [-h] [--gui] [--n-simulations N]
                  [--initial-poses {uniform,uniform-radius,demo-circle,demo-various,load}]
                  [--initial-poses-file INITIAL_POSES_FILE] [--generate-dataset]
                  [--goal-object {station,coloured_station}] [--plots-dataset]
                  [--generate-splits] [--controller CONTROLLER]
                  [--dataset-folder DATASET_FOLDER] [--dataset DATASET]
                  [--models-folder MODELS_FOLDER]
                  [--tensorboard-folder TENSORBOARD_FOLDER] [--model MODEL]
                  [--arch {convnet,convnet_maxpool}] [--loss {mse,smooth_l1}]
                  [--dropout [DROPOUT]] [--train-net] [--evaluate-net]
                  [--compare-models]

Imitation Learning - Task1

optional arguments:
  -h, --help            show this help message and exit
  --gui                 run simulation using the gui (default: False)
  --n-simulations N     number of runs for each simulation (default: 1000)
  --initial-poses {uniform,uniform-radius,demo-circle,demo-various,load}
                        choose how to generate the initial positions for each
                        run, between uniform, uniform-radius, demo-circle,
                        demo-various and load (default: uniform)
  --initial-poses-file INITIAL_POSES_FILE
                        name of the file where to store/load the initial poses
  --generate-dataset    generate the dataset containing the n_simulations
                        (default: False)
  --goal-object {station,coloured_station}
                        choose the type of goal object between station and
                        coloured_station (default: station)
  --plots-dataset       generate the plots of regarding the dataset (default:
                        False)
  --generate-splits     generate dataset splits for training, validation and
                        testing (default: False)
  --controller CONTROLLER
                        choose the controller for the current execution
                        usually between all, learned and omniscient (default:
                        all)
  --dataset-folder DATASET_FOLDER
                        name of the directory containing the datasets
                        (default: datasets)
  --dataset DATASET     choose the datasets to use in the current execution
                        (default: all)
  --models-folder MODELS_FOLDER
                        name of the directory containing the models (default:
                        models)
  --tensorboard-folder TENSORBOARD_FOLDER
                        name of the directory containing Tensorboard logs
                        (default: tensorboard)
  --model MODEL         name of the model (default: net1)
  --arch {convnet,convnet_maxpool}
                        choose the network architecture to use for training
                        the network, between convnet and convnet_maxpool
                        (default: convnet)
  --loss {mse,smooth_l1}
                        choose the loss function to use for training or
                        evaluating the network, between mse and smooth_l1
                        (default: mse)
  --dropout [DROPOUT]   enable dropout after fully-connected layers,
                        optionally setting theprobability of randomly dropping
                        neuron outputs (default: off, default probability:
                        0.5)
  --train-net           train the model (default: False)
  --evaluate-net        generate the plots regarding the model (default:
                        False)
  --compare-models      compare the losses of some models (default: False)
```

##### Tasks

1. Launch file `task1.py` or `task2.py` 