import matplotlib.pyplot as plt

from viz.env import Viz, Env


class _SubplotEnv(Env):
    def __init__(self, gridspec, env):
        self.gridspec = gridspec
        self.env = env

        self.current_subplot = None

    def set_subplot(self, i):
        self.current_subplot = i

    # Env implementation

    def get_figure(self):
        return self.env.get_figure()

    def get_axes(self, *args, **kwargs):
        return self.env.get_axes(*self.gridspec, self.current_subplot, *args, **kwargs)

    @property
    def refresh_interval(self) -> float:
        return self.env.refresh_interval


class GridLayoutViz(Viz):
    def __init__(self, gridspec, vizs, suptitle=None):
        self.gridspec = gridspec
        self.vizs = vizs
        self.suptitle = suptitle

    def _show(self, env: Env):
        env = _SubplotEnv(self.gridspec, env)

        for i, viz in enumerate(self.vizs):
            env.set_subplot(i + 1)
            viz.show(env)

        if self.suptitle:
            fig = env.get_figure()
            fig.suptitle(self.suptitle, fontsize=12, weight='bold')
            self._make_space_above(fig)

    def _make_space_above(self, fig, topmargin=1):
        """
        Increase figure size to make topmargin (in inches) space for titles, without changing the axes sizes
        :param fig:
        :param topmargin:
        """
        fig.tight_layout()

        fig.text(0.07, 0.9, 'Laser scanner response over time', va='top', weight='bold', fontsize=12)
        fig.text(0.58, 0.9, 'Control response over time',       va='top', weight='bold', fontsize=12)

        s = fig.subplotpars
        w, h = fig.get_size_inches()

        fig_h = h - (1 - s.top) * h + topmargin
        fig.subplots_adjust(bottom=s.bottom * h / fig_h, top=1 - topmargin / fig_h)
        fig.set_figheight(fig_h)


    def _update(self):
        for viz in self.vizs:
            viz.update()
