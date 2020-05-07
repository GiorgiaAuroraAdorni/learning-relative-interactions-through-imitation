import sys

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QApplication

import pyenki
from viz import LaserScannerViz

world = pyenki.World(200)

marxbot = pyenki.Marxbot()
marxbot.position = (0, 0)
# marxbot.left_wheel_target_speed = -10
# marxbot.right_wheel_target_speed = 10
world.add_object(marxbot)

marxbot2 = pyenki.Marxbot()
marxbot2.position = (20, -5)
marxbot2.left_wheel_target_speed = 4
marxbot2.right_wheel_target_speed = 4
world.add_object(marxbot2)

cube = pyenki.RectangularObject(10, 10, 15, -1, pyenki.Color(1.0))
cube.position = (-20, 10)
world.add_object(cube)

cylinder = pyenki.CircularObject(10, 15, -1, pyenki.Color(0.0, 1.0, 0.0))
cylinder.position = (20, 20)
world.add_object(cylinder)


app = QApplication.instance()

if not app:
    app = QApplication(sys.argv)
else:
    print("Running within an existing QApplication")

# Create a view --- which will also run ``world.step`` --- and display it
view = pyenki.WorldView(world, run_world_update=True, cam_position=(0, 0),
                        cam_altitude=80, cam_pitch=-np.pi / 2, cam_yaw=np.pi / 2,
                        orthographic=False)
view.show()

# Configure Matplotlib to coexist with the PyEnki Viewer: switch to the Qt5
# backend and enable interactive mode
matplotlib.use("QT5Agg")
plt.ion()

# Create the distance scanner visualization
viz = LaserScannerViz(marxbot)
viz.show()

# Start the event loop
app.exec()
