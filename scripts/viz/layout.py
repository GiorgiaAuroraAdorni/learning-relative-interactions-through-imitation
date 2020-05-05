import matplotlib.pyplot as plt


class GridLayoutViz:
    def __init__(self, gridspec, vizs):
        self.gridspec = gridspec
        self.vizs = vizs

    def show(self, **fig_kw):
        fig = plt.figure(**fig_kw)

        for i, viz in enumerate(self.vizs):
            ax = fig.add_subplot(*self.gridspec, i + 1, **viz.subplot_kw)
            viz.show(ax=ax)

        fig.tight_layout()
