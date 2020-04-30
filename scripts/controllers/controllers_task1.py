from kinematics import wheels_velocities, euclidean_distance, angular_velocity_inplace, angular_velocity, \
    linear_velocity, angle_difference


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
        distance_error = euclidean_distance(state.goal_position, state.position)
        dist_tolerance = 0.1
        angle_error = abs(angle_difference(state.goal_angle, state.angle))
        angle_tolerance = 0.005

        if distance_error > dist_tolerance:
            ang_vel = angular_velocity(state)
            lin_vel = linear_velocity(state, self.max_vel)
        elif angle_error > angle_tolerance:
            ang_vel = angular_velocity_inplace(state)
            lin_vel = 0.0
        else:
            ang_vel = 0.0
            lin_vel = 0.0
            state.goal_reached = True

        left_vel, right_vel = wheels_velocities(lin_vel, ang_vel, self.max_vel)

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
