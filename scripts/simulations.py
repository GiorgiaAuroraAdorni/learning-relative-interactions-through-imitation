import os
import re
from typing import List, Dict

import numpy as np
import pandas as pd
import pyenki
import torch
from tqdm import tqdm

from controllers import controllers_task1
from geometry import Point, Transform
from kinematics import euclidean_distance
from marxbot import MyMarxbot


class GenerateSimulationData:
    OMNISCIENT_CONTROLLER = "omniscient-controller"
    LEARNED_CONTROLLER = r"^learned-controller-net\d"

    @classmethod
    def generate_simulation(cls, runs_dir, n_simulations, controller, args, model_dir=None, model=None):
        """

        :param runs_dir
        :param n_simulations:
        :param controller:
        :param model_dir:
        :param args:
        :param model:
        """
        if controller == cls.OMNISCIENT_CONTROLLER:
            controller_factory = controllers_task1.OmniscientController
        elif re.match(cls.LEARNED_CONTROLLER, controller):
            net = torch.load('%s/%s' % (model_dir, model))

            def controller_factory():
                return controllers_task1.LearnedController(net=net, net_input=args.net_input)
        else:
            raise ValueError("Invalid value for controller")

        world, marxbot, d_object = cls.setup(controller_factory)

        # Generate random polar coordinates to define the area in which the marXbot can spawn, in particular
        # theta ∈ [-π/2, π/2] and r ∈ [d_object.radius/2, max_range * 1.2]
        max_range = 150  # corresponds to the proximity sensors maximal range

        r = np.random.uniform(d_object.radius * 2, max_range * 1.2, n_simulations)
        theta = np.random.uniform(-np.pi / 2, np.pi / 2, n_simulations)
        # The angle is chose randomly in all its possible realisations
        angle = np.random.uniform(0, 2 * np.pi, n_simulations)

        marxbot_rel_poses = np.array([r, theta, angle]).T.reshape(-1, 3)

        dataset_states = pd.DataFrame()
        for n in tqdm(range(n_simulations)):
            try:
                cls.init_positions(marxbot, d_object, marxbot_rel_poses[n])
                run_states = cls.run(marxbot, world, args.gui)

                dataset_run = pd.DataFrame(run_states)
                dataset_run['run'] = n

                dataset_states = dataset_states.append(dataset_run, ignore_index=True)
            except Exception as e:
                print('ERROR: ', e)

        print(dataset_states.goal_reached.value_counts())
        cls.save_dataset(dataset_states, runs_dir)

    @classmethod
    def setup(cls, controller_factory):
        """
        Set up the world.
        :param controller_factory: if the controller is passed, load the learned network
        :return world, marxbot, docking_station
        """
        # Create an unbounded world
        world = pyenki.World()

        marxbot = MyMarxbot(name='marxbot%d' % 1, controller=controller_factory())
        world.add_object(marxbot)

        size = 20.0
        height = 40.0

        d_object = pyenki.CompositeObject(
            [([(0, 1 * size), (0, 0.5 * size), (2 * size, 0.5 * size), (2 * size, 1 * size)], height),
             ([(0, -0.5 * size), (0, -1 * size), (2 * size, -1 * size), (2 * size, -0.5 * size)], height),
             ([(0, 0.5 * size), (0, -0.5 * size), (0.5 * size, -0.5 * size), (0.5 * size, 0.5 * size)], height)],
            -1, pyenki.Color(0, 0.5, 0.5))
        world.add_object(d_object)

        # Decide the pose of the docking_station
        d_object.position = (0, 0)
        d_object.angle = 0

        return world, marxbot, d_object

    @classmethod
    def init_positions(cls, marxbot, d_object, marxbot_rel_pose, min_distance=8.5):
        """
        :param marxbot
        :param d_object
        :param marxbot_rel_pose: initial pose of the marxbot, relative to the goal pose, expressed as r, theta and
                                 angle, that is position in polar coordinates and orientation
        :param min_distance: the minimum distance between the marXbot and any object, that correspond to the radius
                             of the marXbot, is 8.5 cm.
        """
        # The goal pose of the marXbot, defined with respect to the docking station reference frame, has the same y-axis
        # of the docking station and the x-axis translated of the radius of the d_object plus a small arbitrary distance.
        # The goal angle of the marXbot is 180 degree (π).
        increment = d_object.radius + min_distance
        x_goal = d_object.position[0] + increment
        y_goal = d_object.position[1] + 0
        marxbot.goal_position = (x_goal, y_goal)
        marxbot.goal_angle = np.pi

        # Transform the initial pose, relative to the goal, from polar to cartesian coordinates
        r, theta, angle = marxbot_rel_pose
        point_G = Point.from_polar(r, theta)

        # Transform the cartesian coordinates from the goal to the world reference frame
        trasform_D_G = Transform.translate(increment, 0)
        trasform_W_D = Transform.pose_transform(d_object.position, d_object.angle)
        point_W = trasform_W_D @ trasform_D_G @ point_G

        marxbot.initial_position = tuple(Point.to_euclidean(point_W))
        marxbot.position = marxbot.initial_position

        marxbot.initial_angle = angle
        marxbot.angle = marxbot.initial_angle

        marxbot.goal_reached = False

    @classmethod
    def generate_dict(cls, marxbot):
        """
        Save data in a step_state
        :param marxbot
        :return step_state:
        """
        step_state = {
            'name': marxbot.name,
            'initial_position': marxbot.initial_position,
            'initial_angle': marxbot.initial_angle,
            'goal_position': marxbot.goal_position,
            'goal_angle': marxbot.goal_angle,
        }

        return step_state

    @classmethod
    def update_dict(cls, step_state, marxbot):
        """
        Updated data in the step_state instead of rewrite every field to optimise performances
        :param step_state
        :param marxbot
        :return step_state
        """

        step_state['position'] = marxbot.position
        step_state['angle'] = marxbot.angle
        step_state['left_wheel_target_speed'] = marxbot.left_wheel_target_speed
        step_state['right_wheel_target_speed'] = marxbot.right_wheel_target_speed
        step_state['scanner_distances'] = np.array(marxbot.scanner_distances)
        step_state['scanner_image'] = np.array(marxbot.scanner_image)
        step_state['goal_reached'] = marxbot.goal_reached
        step_state['goal_distance'] = euclidean_distance(marxbot.goal_position, marxbot.position)

    @classmethod
    def save_dataset(cls, dataframe, runs_dir):
        """

        :param dataframe:
        :param runs_dir:
        """
        pkl_file = os.path.join(runs_dir, 'simulation.pkl.gz')
        json_file = os.path.join(runs_dir, 'simulation.json.gz')

        dataframe.to_pickle(pkl_file, protocol=4)
        dataframe.to_json(json_file, orient='index')

    @classmethod
    def run(cls, marxbot: MyMarxbot, world: pyenki.World, gui: bool = False, T: float = 15, dt: float = 0.1) -> List[Dict]:
        """
        Run the simulation as fast as possible or using the real time GUI.
        :param marxbot
        :param world
        :param gui
        :param T
        :param dt: update timestep in seconds, should be below 1 (typically .02-.1)
        """

        if gui:
            # We can either run a simulation [in real-time] inside a Qt application
            world.run_in_viewer(cam_position=(100, 0), cam_altitude=350.0, orthographic=True, period=0.1)
        else:
            steps = int(T // dt)

            run_states = []
            step_state = cls.generate_dict(marxbot)

            for s in range(steps):
                world.step(dt)

                cls.update_dict(step_state, marxbot)
                step_state["step"] = s

                run_states.append(step_state.copy())

                # Check if the robot has reached the target
                if marxbot.goal_reached:
                    break

            return run_states
