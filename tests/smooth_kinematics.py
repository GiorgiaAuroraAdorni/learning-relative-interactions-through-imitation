import matplotlib
import matplotlib.pyplot as plt
import numpy as np
## Setup the simulation
import pyenki
from PyQt5.QtWidgets import QApplication

import controllers.controllers_task1 as controllers
import viz
from marxbot import MyMarxbot

world = pyenki.World(200)

controller = controllers.OmniscientController()
marxbot = MyMarxbot("marxbot", controller)
marxbot.position = (0, 0)
marxbot.angle = 0
marxbot.goal_position = (38, 0)
marxbot.goal_angle = np.pi
world.add_object(marxbot)

size = 20.0
height = 40.0

d_object = pyenki.CompositeObject(
    [([(0, 1 * size), (0, 0.5 * size), (2 * size, 0.5 * size), (2 * size, 1 * size)], height),
     ([(0, -0.5 * size), (0, -1 * size), (2 * size, -1 * size), (2 * size, -0.5 * size)], height),
     ([(0, 0.5 * size), (0, -0.5 * size), (0.5 * size, -0.5 * size), (0.5 * size, 0.5 * size)], height)],
    -1, pyenki.Color(0, 0.5, 0.5))
world.add_object(d_object)

# Decide the pose of the docking_station
d_object.position = (0, 0)
d_object.angle = 0

## Start the application
app = QApplication.instance()

if not app:
    pyenki.createApp()
else:
    print("Running within an existing QApplication")

# Configure Matplotlib to coexist with the PyEnki Viewer: switch to the Qt5
# backend and enable interactive mode
matplotlib.use("QT5Agg")
plt.ion()

# Create a view --- which will also run ``world.step`` --- and display it
view = pyenki.WorldView(world, run_world_update=True,
                        cam_position=(0, 0), cam_altitude=350.0, cam_pitch=-np.pi / 2, cam_yaw=np.pi / 2,
                        orthographic=False, period=0.06)
view.show()

# Create the visualizations
env = viz.FuncAnimationEnv([
    viz.GridLayout((1, 3), [
        viz.TrajectoryViz(marxbot),
        viz.LaserScannerViz(marxbot),
        viz.ControlSignalsViz(marxbot)
    ], suptitle='Smooth Kinematics Demo')
], refresh_interval=0.1)
env.show(figsize=(14, 4))

# Start the event loop
pyenki.execApp(view)
