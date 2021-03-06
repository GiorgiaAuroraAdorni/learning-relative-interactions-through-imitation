import os
from datetime import datetime


def check_dir(directory):
    """
    Check if the path is a directory, if not create it.
    :param directory: path to the directory
    """
    os.makedirs(directory, exist_ok=True)


def directory_for_dataset(dataset, controller):
    """

    :param dataset:
    :param controller:
    :return run_dir, run_img_dir, run_video_dir:
    """
    run_dir = os.path.join(dataset, controller)

    run_img_dir = os.path.join(run_dir, 'images')
    check_dir(run_img_dir)

    run_video_dir = os.path.join(run_dir, 'videos')
    check_dir(run_video_dir)

    return run_dir, run_img_dir, run_video_dir


def directory_for_model(args):
    """

    :param args:
    :return:
    """
    model_dir = os.path.join(args.models_folder, args.model)

    model_img_dir = os.path.join(model_dir, 'images')
    check_dir(model_img_dir)

    model_video_dir = os.path.join(model_dir, 'videos')
    check_dir(model_video_dir)

    metrics_path = os.path.join(model_dir, 'metrics.pkl')

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    tboard_dir = os.path.join(args.tensorboard_folder, args.model, run_name)
    check_dir(tboard_dir)

    return model_dir, model_img_dir, model_video_dir, metrics_path, tboard_dir


def unpack(dataset, dim):
    """
    Unpack a `xr.Dataset` along a given dimension

    :param dataset: dataset object to be unpacked
    :param dim: dimension along which the dataset should be unpacked
    :return: sequence of one `xr.Dataset`s for each value of dim
    """
    return (dataset.loc[{dim: value}] for value in dataset[dim])


def unpack_tuple(x):
    """

    :param x:
    :return:
    """
    if len(x) == 1:
        return x[0]
    else:
        return x
