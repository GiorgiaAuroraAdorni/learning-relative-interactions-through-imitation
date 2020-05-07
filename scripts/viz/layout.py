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

        if self.suptitle:
            fig.suptitle(self.suptitle, fontsize=12, weight='bold')

    def _update(self):
        for viz in self.vizs:
            viz.update()
