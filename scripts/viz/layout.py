from matplotlib import pyplot as plt

from viz.env import Viz, Env, PassthroughEnv


class _SubplotEnv(PassthroughEnv):
    def __init__(self, gridspec, env, sharey):
        super().__init__(env)

        self.gridspec = gridspec
        self.sharey = sharey

        self.axes = []
        self.current_subplot = None

    def set_subplot(self, i):
        self.current_subplot = i

    # Env implementation

    def get_axes(self, *args, **kwargs):
        if self.sharey and len(self.axes) > 0:
            sharey = self.axes[0]
        else:
            sharey = None

        axes = super().get_axes(*self.gridspec, self.current_subplot, *args, sharey=sharey, **kwargs)
        self.axes.append(axes)

        return axes


class GridLayout(Viz):
    def __init__(self, gridspec, vizs, suptitle=None, sharey=False):
        self.gridspec = gridspec
        self.vizs = vizs
        self.suptitle = suptitle
        self.sharey = sharey

    def _show(self, env: Env):
        env = _SubplotEnv(self.gridspec, env, sharey=self.sharey)
        fig = env.get_figure()

        for i, viz in enumerate(self.vizs):
            env.set_subplot(i + 1)
            viz.show(env)

        if self.suptitle:
            fig.suptitle(self.suptitle, fontsize=12, weight='bold')

    def _update(self):
        for viz in self.vizs:
            viz.update()
