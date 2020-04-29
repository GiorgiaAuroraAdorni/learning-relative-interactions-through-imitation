import re

import numpy as np
import pyenki
import torch
from tqdm import tqdm

from controllers import controllers_task1
from marxbot import MyMarxbot
from geometry import Point, Transform


class GenerateSimulationData:
    OMNISCIENT_CONTROLLER = "omniscient-controller"
    LEARNED_CONTROLLER = r"^learned-controller-net\d"

    @classmethod
    def generate_simulation(cls, simulations, controller, args, model_dir=None, model=None):
        """

        :param simulations:
        :param controller:
        :param model_dir:
        :param args:
        :param model:
        """
        if controller == cls.OMNISCIENT_CONTROLLER:
            controller_factory = controllers_task1.OmniscientController
        elif re.match(cls.LEARNED_CONTROLLER, controller):
            net = torch.load('%s/%s' % (model_dir, model))

            def controller_factory(**kwargs):
                return controllers_task1.LearnedController(net=net, net_input=args.net_input, **kwargs)
        else:
            raise ValueError("Invalid value for controller")

        world, marxbot, d_object = cls.setup(controller_factory)

        # Generate random polar coordinates to define the area in which the marXbot can spawn, in particular
        # theta ∈ [-π/2, π/2] and r ∈ [d_object.radius/2, maximum_gap *2]
        maximum_gap = 150  # corresponds to the proximity sensors maximal range

        r = np.random.uniform(d_object.radius * 2, maximum_gap * 2, simulations)
        theta = np.random.uniform(-np.pi / 2, np.pi / 2, simulations)
        # The angle is chose randomly in all its possible realisations
        angle = np.random.uniform(0, 2 * np.pi, simulations)

        marxbot_distances = np.array([r, theta, angle]).T.reshape(-1, 3)

        for s in tqdm(range(simulations)):
            try:
                cls.init_positions(marxbot, d_object, marxbot_distances, s)
                cls.run(args.gui)
            except Exception as e:
                print('ERROR: ', e)

    @classmethod
    def setup(cls, controller_factory):
        """
        Set up the world FIXME how.
        :param controller_factory: if the controller is passed, load the learned network
        :return world, marxbot, docking_station
        """
        # FIXME
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

        # Decide the pose of the d_object
        d_object.position = (0, 0)
        d_object.angle = 0

        return world, marxbot, d_object

    @classmethod
    def init_positions(cls, marxbot, d_object, distances, simulation, min_distance=8.5):
        """
        :param marxbot
        :param d_object
        :param distances: contains r, theta and angle that are the random polar coordinates for the marxbot
        :param simulation
        :param min_distance: the minimum distance between the marXbot and any object, that correspond to the radius
                             of the marXbot, is 8.5 cm.
        """
        # The goal pose of the marXbot is a on the same y-axis of the d_object and with the x-axis translated of the
        # radius of the d_object plus a small arbitrary distance, with respect to the d_object.
        # The goal angle of the marXbot is 180 degree (π).
        increment = d_object.radius + min_distance
        x_goal = d_object.position[0] + increment
        y_goal = d_object.position[1] + 0
        marxbot.goal_position = (x_goal, y_goal)
        marxbot.goal_angle = np.pi

        # Transform the distance from polar coordinates to cartesian coordinates
        r, theta, angle = distances[simulation]
        point_G = Point.from_polar(r, theta)

        # Transform the cartesian coordinates from the goal to the world reference frame
        trasform_W_D = Transform.rotate(d_object.angle) @ Transform.translate(*d_object.position)
        trasform_D_G = Transform.translate(increment, 0)
        point_W = trasform_W_D @ trasform_D_G @ point_G

        marxbot.initial_position = tuple(Point.to_euclidean(point_W))
        marxbot.position = marxbot.initial_position

        marxbot.initial_angle = angle
        marxbot.angle = marxbot.initial_angle

        marxbot.dictionary = None

    @classmethod
    def run(cls, world: pyenki.World, gui: bool = False, T: float = 2, dt: float = 0.1, tol: float = 0.1) -> None:
        """
        Run the simulation as fast as possible or using the real time GUI.
        :param world
        :param gui
        :param T
        :param dt: update timestep in seconds, should be below 1 (typically .02-.1)
        :param tol: tolerance used to verify if the robot reaches the target
        """

        if gui:
            # We can either run a simulation [in real-time] inside a Qt application
            world.run_in_viewer(cam_position=(100, 0), cam_altitude=350.0, orthographic=True, period=0.1)
        else:
            steps = int(T // dt)

            for s in range(steps):
                world.step(dt)
