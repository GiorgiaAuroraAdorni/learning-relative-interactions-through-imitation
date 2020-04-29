import pyenki

from controllers import controllers_task1


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

        self.dictionary = None

    def controlStep(self, dt: float) -> None:
        """
        Perform one control step:
        Move the robots in such a way they stand face a horseshoe-shaped object.
        It is possible to use the omniscient or the learned controller.
        :param dt: control step duration
        """

        left_vel, right_vel = self.controller.perform_control(self, dt)

        self.left_wheel_target_speed = left_vel
        self.right_wheel_target_speed = right_vel
