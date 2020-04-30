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


def linear_velocity(state, min_vel, max_vel, constant=4):
    """
    :param state
    :param constant
    :return: linear velocity
    """
    # velocity = constant * euclidean_distance(state)
    velocity = constant * signed_distance(state)
    velocity = min(max(min_vel/1.5, velocity), max_vel/1.5)
    return velocity


def angular_difference(state):
    """
    :param state:
    :return: the difference between the current angle and the goal angle
    """
    return np.arctan2(np.sin(state.goal_angle - state.agle), np.cos(state.goal_angle - state.agle))


def angular_velocity_rotation(state, constant=1):
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


def angular_velocity(state, constant=4):
    """
    :param state:
    :param constant:
    :return: angular velocity
    """
    return constant * np.arctan2(np.sin(steering_angle(state) - state.angle),
                                 np.cos(steering_angle(state) - state.angle))


def get_angle(state):
    """Returns angle of the vector from pose to goal.
    :param state
    :return:
    """
    goal_x, goal_y = state.goal_position
    pose_x, pose_y = state.position

    return np.arctan2(goal_y - pose_y, goal_x - pose_x)


def icr_angular_velocity(state):
    """
    :param state:
    :return: angular velocity
    """
    # compute angle of vector from turtle to goal
    angle = get_angle(state)
    delta_angle = angle - state.angle

    return np.sin(delta_angle)


def wheels_velocities(state, min_vel=-np.inf, max_vel=np.inf, wheel_distance=15):
    """

    :param state:
    :param min_vel
    :param max_vel
    :param wheel_distance:
    :return left_wheel_target_speed, right_wheel_target_speed
    """

    # ang_vel = angular_velocity(state)
    ang_vel = icr_angular_velocity(state)
    lin_vel = linear_velocity(state, min_vel, max_vel)

    left_wheel_target_speed = lin_vel + wheel_distance * ang_vel
    right_wheel_target_speed = lin_vel - wheel_distance * ang_vel

    left_wheel_target_speed = min(max(min_vel, left_wheel_target_speed), max_vel)
    right_wheel_target_speed = min(max(min_vel, right_wheel_target_speed), max_vel)

    return left_wheel_target_speed, right_wheel_target_speed
