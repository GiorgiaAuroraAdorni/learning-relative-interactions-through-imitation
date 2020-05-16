from abc import ABC, abstractmethod

import numpy as np
import torch

from kinematics import to_wheels_velocities, euclidean_distance, angle_difference, steering_angle


class Controller(ABC):

    def __init__(self):
        self.max_vel = 30

        self.distance_tol = 0.1
        self.angle_tol = 0.005

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
        velocities = self._perform_control(state, dt)

        # Detect when the robot reaches the goal
        distance_error = euclidean_distance(state.goal_position, state.position)
        angle_error = angle_difference(state.goal_angle, state.angle)

        goal_reached = (distance_error < self.distance_tol and angle_error < self.angle_tol)

        return velocities, goal_reached

    @abstractmethod
    def _perform_control(self, state, dt):
        pass


class OmniscientController(Controller):
    """
    The robots can be moved following an optimal “omniscient” controller. In this case, based on the poses of
    the robots, the omniscient control moves the robots at a constant speed, calculating the distance from the
    actual pose to the target one.
    """
    def __init__(self):
        super().__init__()

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
    def reference_heading(theta, k1=1):
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
    def linear_velocity(max_vel, k, beta=0.4, lambd=2):
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

    def _perform_control(self, state, dt):
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
        delta_hat = self.reference_heading(theta)
        k = self.curvature(r, theta, delta, delta_hat)

        lin_vel = self.linear_velocity(min(self.max_vel, 2 * r), k)
        ang_vel = self.angular_velocity(k, lin_vel)
        left_vel, right_vel = to_wheels_velocities(lin_vel, ang_vel)

        return left_vel, right_vel


class LearnedController(Controller):
    """
    The robots can be moved following a controller learned by a neural network.
    """

    def __init__(self, net):
        super().__init__()

        self.net = net
        if self.net is None:
            raise ValueError("Value for net not provided")

    @staticmethod
    def input_to_tensor(state):
        """
        :param state
        :return input:
        """
        scanner_image = state.scanner_image
        scanner_distances = state.scanner_distances

        # Add a new 'channels' dimension to scanner_distances so it can be
        # concatenated with scanner_image
        scanner_distances = np.expand_dims(scanner_distances, axis=-1)

        # Concatenate the two variables to a single array and transpose the dimensions
        # to match the PyTorch convention of samples ⨉ channels ⨉ angles
        scanner_data = np.concatenate([scanner_image, scanner_distances], axis=-1)
        scanner_data = np.expand_dims(np.transpose(scanner_data), 0)

        # FIXME: maybe save directly as float32?
        input = torch.as_tensor(scanner_data, dtype=torch.float)

        return input

    def _perform_control(self, state, dt):
        """
        Extract the input sensing from the scanner_image and scanner_distances
        readings.
        Generate the output speed using the learned controller.

        :param state
        :param dt
        """
        input = self.input_to_tensor(state)

        velocities = torch.squeeze(self.net(input))
        left_vel, right_vel = velocities.detach().tolist()

        return left_vel, right_vel
