import sys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QApplication


## Setup the simulation
import pyenki

from marxbot import MyMarxbot
from viz.scanner import DistanceScannerViz
import controllers.controllers_task1 as controllers

world = pyenki.World(200)

controller = controllers.OmniscientController()
marxbot = MyMarxbot("marxbot", controller)
marxbot.position = (0, 0)
marxbot.angle = 0
marxbot.goal_position = (0, 20)
marxbot.goal_angle = np.pi/2
world.add_object(marxbot)


## Start the application
app = QApplication.instance()

if not app:
    pyenki.createApp()
else:
    print("Running within an existing QApplication")

# Configure Matplotlib to coexist with the PyEnki Viewer: switch to the Qt5
# backend and enable interative mode
matplotlib.use("QT5Agg")
plt.ion()

# Create a view --- which will also run ``world.step`` --- and display it
view = pyenki.WorldView(world, run_world_update=True, cam_position=(0, 0),
                        cam_altitude=80, cam_pitch=-np.pi / 2, cam_yaw=np.pi / 2,
                        orthographic=False)
view.show()

# Create the distance scanner visualization
viz = DistanceScannerViz(marxbot)

# Start the event loop
pyenki.execApp(view)
