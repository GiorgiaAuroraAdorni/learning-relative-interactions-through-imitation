import numpy as np


def euclidean_distance(state):
    """
    :return: Euclidean distance between current pose and the goal pose
    """
    return np.sqrt(pow((state.goal_position[0] - state.position[0]), 2) +
                   pow((state.goal_position[1] - state.position[1]), 2))


def signed_distance(state):
    """
    :return: Signed distance between current and the goal position, along the current theta of the robot
    """
    a = state.position[0] * np.cos(state.angle) + state.position[1] * np.sin(state.angle)
    b = state.goal_position[0] * np.cos(state.angle) + state.goal_position[1] * np.sin(state.angle)

    return b - a


def linear_velocity(state, constant=10):
    """
    :param state
    :param constant
    :return: linear velocity
    """
    velocity = constant * signed_distance(state)
    return velocity


def angular_difference(state):
    """
    :param state:
    :return: the difference between the current angle and the goal angle
    """
    return np.arctan2(np.sin(state.goal_angle - state.agle), np.cos(state.goal_angle - state.agle))


def angular_velocity_rotation(state, constant=12):
    """
    :param state:
    :param constant:
    :return: the angular velocity computed using the angle difference
    """
    return constant * angular_difference(state)


def steering_angle(state):
    """
    :param state:
    :return: steering angle
    """
    return np.arctan2(state.goal_position[1] - state.position[1], state.goal_position[0] - state.position[0])


def angular_velocity(state, constant=8):
    """
    :param state:
    :param constant:
    :return: angular velocity
    """
    return constant * np.arctan2(np.sin(steering_angle(state) - state.angle),
                                 np.cos(steering_angle(state) - state.angle))


def wheels_velocities(state, min_vel=-np.inf, max_vel=np.inf, wheel_distance=15):
    """

    :param state:
    :param min_vel
    :param max_vel
    :param wheel_distance:
    :return left_wheel_target_speed, right_wheel_target_speed
    """
    ang_vel = angular_velocity(state)
    lin_vel = linear_velocity(state)

    left_wheel_target_speed = + (wheel_distance * ang_vel) + lin_vel
    right_wheel_target_speed = - (wheel_distance * ang_vel) + lin_vel

    left_wheel_target_speed = min(max(min_vel, left_wheel_target_speed), max_vel)
    right_wheel_target_speed = min(max(min_vel, right_wheel_target_speed), max_vel)

    return left_wheel_target_speed, right_wheel_target_speed
