import numpy as np

from kinematics import to_wheels_velocities, euclidean_distance, angle_difference, steering_angle

class Controller:

    def perform_control(self, state, dt):
        """
        Move the robots using the omniscient controller by setting the target {left,right} wheel speed
        each at the same value in order to moves the robot straight ahead.
        The speed is computed as follow:
            velocity = constant * self.signed_distance()
        where the constant is set to 4 and the signed_distance is the distance between the current and the goal
        position of the robot, along the current theta of the robot.
        :param state
        :param dt
        """
        raise NotImplementedError("The subclass should implement this method")


class OmniscientController(Controller):
    """
    The robots can be moved following an optimal “omniscient” controller. In this case, based on the poses of
    the robots, the omniscient control moves the robots at a constant speed, calculating the distance from the
    actual pose to the target one.
    """
    def __init__(self):
        super().__init__()

        self.max_vel = 30

        self.distance_tol = 0.1
        self.angle_tolerance = 0.005

    @staticmethod
    def current_state(state):
        """

        :param state:
        :return r, theta: polar coordinates of the target pose with respect to the actual position
        """
        steer_angle = steering_angle(state)

        r = euclidean_distance(state.goal_position, state.position)
        theta = angle_difference(steer_angle, state.goal_angle)
        delta = angle_difference(steer_angle, state.angle)

        return r, theta, delta

    @staticmethod
    def virtual_controller(theta, k1=1):
        """

        :param theta:
        :param k1: FIXME
        :return delta_hat:
        """
        delta_hat = np.arctan(-k1 * theta)

        return delta_hat

    @staticmethod
    def curvature(r, theta, delta, delta_hat, k1=1, k2=3):
        """

        :param state:
        :param r:
        :param theta:
        :param delta:
        :param delta_hat:
        :param k1:
        :param k2:
        :return k:
        """
        z = delta - delta_hat
        k = (1 / r) * (k2 * z + (1 + (k1 / (1 + pow(k1 * theta, 2)))) * np.sin(delta))
        return k

    @staticmethod
    def linear_velocity(max_vel, k, beta=1, lambd=1):
        """

        :param max_vel:
        :param k: curvature
        :param beta:
        :param lambd:
        :return lin_vel:
        """
        lin_vel = max_vel / (1 + beta * pow(abs(k), lambd))

        return lin_vel

    @staticmethod
    def angular_velocity(k, lin_vel):
        """

        :param k:
        :param lin_vel:
        :return:
        """
        ang_vel = k * lin_vel

        return ang_vel

    def perform_control(self, state, dt):
        """
        Move the robots using the omniscient controller by setting the target {left,right} wheel speed
        each at the same value in order to moves the robot straight ahead.
        The speed is computed as follow:
            velocity = constant * self.signed_distance()
        where the constant is set to 4 and the signed_distance is the distance between the current and the goal
        position of the robot, along the current theta of the robot.
        :param state
        :param dt
        """

        r, theta, delta = self.current_state(state)
        delta_hat = self.virtual_controller(theta)
        k = self.curvature(r, theta, delta, delta_hat)

        lin_vel = self.linear_velocity(min(self.max_vel, 2 * r), k)
        ang_vel = self.angular_velocity(k, lin_vel)
        left_vel, right_vel = to_wheels_velocities(lin_vel, ang_vel)

        # Detect when the robot reaches the goal
        distance_error = r
        # Equivalent to state.goal_angle - state.angle
        angle_error = abs(delta - theta)

        if distance_error < self.distance_tol and angle_error < self.angle_tolerance:
            state.goal_reached = True

        return left_vel, right_vel


class LearnedController(Controller):
    """
    The robots can be moved following a controller learned by a neural network.
    """

    def __init__(self, net, net_input):
        super().__init__()

        self.net = net
        if self.net is None:
            raise ValueError("Value for net not provided")

        self.net_controller = net.controller()
        self.net_input = net_input

    def perform_control(self, state, dt):
        """
        Extract the input sensing from the list of (7) proximity sensor readings, one for each sensors.
        The first 5 entries are from frontal sensors ordered from left to right.
        The last two entries are from rear sensors ordered from left to right.
        Then normalise each value of the list, by dividing it by 1000.

        Generate the output speed using the learned controller.

        Move the robots not to the end of the line using the controller, setting the target {left,right} wheel speed
        each at the same value in order to moves the robot straight ahead.
        :param state
        :param dt
        """
        sensing = get_input_sensing(self.net_input, state)

        speed = float(self.net_controller(sensing)[0])

        if state.initial_position[0] != state.goal_position[0]:
            return speed
        else:
            return 0
