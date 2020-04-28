import os
import numpy as np


def check_dir(directory):
    """
    Check if the path is a directory, if not create it.
    :param directory: path to the directory
    """
    os.makedirs(directory, exist_ok=True)


def signed_distance(state):
    """
    :return: Signed distance between current and the goal position, along the current theta of the robot
    """
    a = state.position[0] * np.cos(state.angle) + state.position[1] * np.sin(state.angle)
    b = state.goal_position[0] * np.cos(state.angle) + state.goal_position[1] * np.sin(state.angle)

    return b - a


def euclidean_distance(state):
    """
    :return: Euclidean distance between current pose and the goal pose
    """
    return np.sqrt(pow((state.goal_position[0] - state.position[0]), 2) +
                   pow((state.goal_position[1] - state.position[1]), 2))


def linear_vel(state, constant=10):
    """
    :param state
    :param constant
    :return: linear velocity
    """
    velocity = constant * signed_distance(state)
    return velocity


def steering_angle(state):
    """
    :param state:
    :return: steering angle
    """
    return np.arctan2(state.goal_position[1] - state.position[1], state.goal_position[0] - state.position[0])


def angular_vel(state, constant=8):
    """
    :param state:
    :param constant:
    :return: angular velocity
    """
    # return constant * (self.steering_angle(goal_pose) - self.pose.theta)
    return constant * np.arctan2(np.sin(steering_angle(state) - state.angle),
                                 np.cos(steering_angle(state) - state.angle))


def angle_difference(state):
    """
    :param state:
    :return: the difference between the current angle and the goal angle
    """
    return np.arctan2(np.sin(state.goal_angle - state.agle), np.cos(state.goal_angle - state.agle))


def angular_vel_rot(state, constant=12):
    """
    :param state:
    :param constant:
    :return: the angular velocity computed using the angle difference
    """
    return constant * angle_difference(state)