import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtCore import QObject


class ControllerViz(QObject):
    def __init__(self, marxbot, refresh_interval=30):
        super().__init__()

        self.marxbot = marxbot

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.startTimer(refresh_interval)

    def timerEvent(self, event):
        self.fig.canvas.draw_idle()

