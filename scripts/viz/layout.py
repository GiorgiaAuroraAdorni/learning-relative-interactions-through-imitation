from viz.env import Viz, Env, PassthroughEnv


class _SubplotEnv(PassthroughEnv):
    def __init__(self, gridspec, env):
        super().__init__(env)

        self.gridspec = gridspec
        self.current_subplot = None

    def set_subplot(self, i):
        self.current_subplot = i

    # Env implementation

    def get_axes(self, *args, **kwargs):
        return super().get_axes(*self.gridspec, self.current_subplot, *args, **kwargs)


class GridLayoutViz(Viz):
    def __init__(self, gridspec, vizs, suptitle=None):
        self.gridspec = gridspec
        self.vizs = vizs
        self.suptitle = suptitle

    def _show(self, env: Env):
        env = _SubplotEnv(self.gridspec, env)
        fig = env.get_figure()

        for i, viz in enumerate(self.vizs):
            env.set_subplot(i + 1)
            viz.show(env)

        # Ensure that the layout of the subplots is consistent
        fig.tight_layout()

        if self.suptitle:
            fig.suptitle(self.suptitle, fontsize=12, weight='bold')
            self._make_space_above(fig)

    def _make_space_above(self, fig, topmargin=1):
        """
        Increase figure size to make topmargin (in inches) space for titles, without changing the axes sizes
        :param fig:
        :param topmargin:
        """
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
