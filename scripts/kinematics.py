import numpy as np


def euclidean_distance(position1, position2):
    """
    :return: Euclidean distance between current pose and the goal pose
    """
    x1, y1 = position1
    x2, y2 = position2

    return np.sqrt(pow((x2 - x1), 2) +
                   pow((y2 - y1), 2))


def signed_distance(state):
    """
    :return: Signed distance between current and the goal position, along the current theta of the robot
    """
    a = state.position[0] * np.cos(state.angle) + state.position[1] * np.sin(state.angle)
    b = state.goal_position[0] * np.cos(state.angle) + state.goal_position[1] * np.sin(state.angle)

    return b - a


def linear_velocity(state, max_vel, constant=8):
    """
    :param state
    :param max_vel
    :param constant
    :return: linear velocity
    """
    velocity = constant * signed_distance(state)
    velocity = np.clip(velocity, -max_vel/1.2, max_vel/1.2)

    return velocity


def angle_difference(alpha, beta):
    """
    :return: the difference between two angles alpha and beta.
    """
    return np.arctan2(np.sin(alpha - beta), np.cos(alpha - beta))


def angular_velocity_inplace(state, constant=1):
    """
    :param state:
    :param constant:
    :return: the angular velocity computed using the angle difference
    """
    return constant * angle_difference(state.goal_angle, state.angle)


def steering_angle(state):
    """Returns angle of the vector from pose to goal.
    :param state
    :return:
    """
    goal_x, goal_y = state.goal_position
    pose_x, pose_y = state.position

    return np.arctan2(goal_y - pose_y, goal_x - pose_x)


def angular_velocity(state, constant=1):
    """
    :param state:
    :param constant
    :return: angular velocity
    """
    delta_angle = angle_difference(steering_angle(state), state.angle)
    # Extract perpendicular component to handle the case when the robot orientation is 0 and to reach the target
    # position it is necessary to go backward.
    delta_angle = np.sin(delta_angle)

    return constant * delta_angle


def new_linear_velocity(max_vel, k, beta=1, lambd=1):
    """

    :param max_vel:
    :param k: curvature
    :param beta:
    :param lambd:
    :return lin_vel:
    """
    lin_vel = max_vel / (1 + beta * pow(abs(k), lambd))

    return lin_vel


def new_angular_velocity(k, lin_vel):
    """

    :param k:
    :param lin_vel:
    :return:
    """
    ang_vel = k * lin_vel

    return ang_vel


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
