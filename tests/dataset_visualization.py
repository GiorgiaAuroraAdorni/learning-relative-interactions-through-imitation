import os
import numpy as np

import viz
from dataset import load_dataset

# Load the dataset
controller = "omniscient-controller"
datasets_dir = "datasets/"
runs_dir = os.path.join(datasets_dir, controller)
dataset = load_dataset(runs_dir)

sample_id = np.random.choice(dataset.sizes['sample'])
run_id = dataset.run[sample_id]
run = dataset.where(dataset.run == run_id, drop=True)

print("Generating video for run %d…" % run_id)

marxbot = viz.DatasetSource(run)

# Create the visualizations
env = viz.FuncAnimationEnv([
    viz.GridLayout((1, 2), [
        viz.LaserScannerViz(marxbot),
        viz.ControlSignalsViz(marxbot)
    ], suptitle="%s: run %d" % (controller, run_id))
], sources=[marxbot])
env.show(figsize=(9, 4))

video_path = os.path.join(runs_dir, 'run-%d.mp4' % run_id)
env.save(video_path, dpi=300)
