import pyenki


class MyMarxbot(pyenki.Marxbot):
    """
    Superclass: `pyenki.Marxbot` -> the world update step will automatically call the Marxbot `controlStep`.
    """

    def __init__(self, name, controller, **kwargs) -> None:
        """
        :param name
        :param controller
        :param kwargs
        """
        super().__init__(**kwargs)

        self.name = name
        self.controller = controller

        self.initial_position = None
        self.initial_angle = None

        self.goal_position = None
        self.goal_angle = None

        self.goal_reached = False

    def controlStep(self, dt: float) -> None:
        """
        Perform one control step:
        Move the robots in such a way they stand face a horseshoe-shaped object.
        It is possible to use the omniscient or the learned controller.
        :param dt: control step duration
        """

        (left_vel, right_vel), goal_reached = self.controller.perform_control(self, dt)

        self.left_wheel_target_speed = left_vel
        self.right_wheel_target_speed = right_vel
        self.goal_reached = goal_reached

    @property
    def wheel_target_speeds(self):
        return [self.left_wheel_target_speed, self.right_wheel_target_speed]
