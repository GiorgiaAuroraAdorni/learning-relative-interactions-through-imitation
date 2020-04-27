import math
import threading
#
# import matplotlib
# matplotlib.use("QT5Agg")

import numpy as np
import matplotlib.pyplot as plt

from PyQt5 import QtCore

import pyenki

world = pyenki.World(100)

marxbot = pyenki.Marxbot()
marxbot.position = (0, 0)
marxbot.left_wheel_target_speed = -10
marxbot.right_wheel_target_speed = 10
world.add_object(marxbot)

marxbot2 = pyenki.Marxbot()
marxbot2.position = (20, 5)
world.add_object(marxbot2)

cube = pyenki.RectangularObject(10, 10, 15, -1, pyenki.Color(1.0))
cube.position = (-20, 5)
world.add_object(cube)


class Test(pyenki.Thymio2):
    def __init__(self):
        super().__init__()

        plt.ion()

        self.fig = plt.figure()

        self.updateRate = 0.060
        self.elapsedTime = 0

    def controlStep(self, dt):
        if self.elapsedTime < self.updateRate:
            self.elapsedTime += dt
            return

        self.elapsedTime = 0

        self.fig.clear()
        plt.subplot(2, 1, 1, polar=True)
        plt.plot(np.linspace(-np.pi, np.pi, 180),
                np.minimum(np.sqrt(marxbot.scanner_distances), 50))

        image = np.array(marxbot.scanner_image)[np.newaxis, ...]
        plt.subplot(2, 1, 2)
        plt.imshow(image, aspect=10)
        plt.axis("off")


test = Test()
world.add_object(test)

# Check if a QtApplication is running
if not QtCore.QCoreApplication.instance():
    print('No QtApplication active')
    world.run_in_viewer(cam_position=(0, 0), cam_altitude=80,
                        cam_pitch=-math.pi / 2, cam_yaw=math.pi / 2,
                        orthographic=False)
else:
    # Create a view --- which will also run ``world.step`` --- and display it
    view = pyenki.WorldView(world, run_world_update=True, cam_position=(0, 0),
                            cam_altitude=80, cam_pitch=-math.pi / 2, cam_yaw=math.pi / 2,
                            orthographic=False)
    view.show()
