import numpy as np


def euclidean_distance(position1, position2):
    """
    :return: Euclidean distance between current pose and the goal pose
    """
    x1, y1 = position1
    x2, y2 = position2

    return np.sqrt(pow((x2 - x1), 2) +
                   pow((y2 - y1), 2))


def angle_difference(alpha, beta):
    """
    :return: the difference between two angles alpha and beta.
    """
    return np.arctan2(np.sin(alpha - beta), np.cos(alpha - beta))


def steering_angle(state):
    """Returns angle of the vector from pose to goal.
    :param state
    :return:
    """
    goal_x, goal_y = state.goal_position
    pose_x, pose_y = state.position

    return np.arctan2(goal_y - pose_y, goal_x - pose_x)


def to_wheels_velocities(lin_vel, ang_vel, wheel_distance=15):
    """
    :param lin_vel
    :param ang_vel
    :param wheel_distance:
    :return left_speed, right_speed
    """

    left_speed  = lin_vel - wheel_distance * ang_vel
    right_speed = lin_vel + wheel_distance * ang_vel

    return left_speed, right_speed


def to_robot_velocities(left_speed, right_speed, wheel_distance=15):
    lin_vel = ( left_speed + right_speed) / 2
    ang_vel = (right_speed -  left_speed) / (2 * wheel_distance)

    return lin_vel, ang_vel
