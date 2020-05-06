import os

import numpy as np
import xarray as xr

from dataset import load_dataset

from viz.controller import ControllerViz
from viz.env import FuncAnimationEnv
from viz.layout import GridLayoutViz
from viz.scanner import DistanceScannerViz

# Load the dataset
runs_dir = "datasets/omniscient-controller"
dataset = load_dataset(runs_dir)

sample_id = np.random.choice(dataset.sizes['sample'])
run_id = dataset.run[sample_id]
run = dataset.where(dataset.run == run_id, drop=True)

print("Generating video for run %d…" % run_id)


class AnimationDataset:
    def __init__(self, dataset: xr.Dataset, dim="sample"):
        self.dataset = dataset
        self.dim = dim

        self.current_sample = None

    def update(self, frame):
        self.current_sample = self.dataset[{self.dim: frame}]

    def __len__(self):
        return self.dataset.sizes[self.dim]

    def __getattr__(self, name):
        return getattr(self.current_sample, name)


marxbot = AnimationDataset(run)

# Create the visualizations
env = FuncAnimationEnv([
    GridLayoutViz((1, 2), [
        DistanceScannerViz(marxbot),
        ControllerViz(marxbot)
    ])
], datasets=[marxbot])
env.show(figsize=(9, 4))

video_path = os.path.join(runs_dir, 'run-%d.mp4' % run_id)
env.save(video_path, dpi=300)
