import re

import numpy as np
import pyenki
import torch
from tqdm import tqdm

from controllers import controllers_task1
from dataset import DatasetBuilder
from geometry import Point, Transform
from kinematics import euclidean_distance, angle_difference
from marxbot import MyMarxbot


class GenerateSimulationData:
    OMNISCIENT_CONTROLLER = "omniscient-controller"
    LEARNED_CONTROLLER = r"^learned-controller-net\d"

    @classmethod
    def generate_simulation(cls, n_simulations, controller, args, model_dir=None, model=None):
        """

        :param n_simulations:
        :param controller:
        :param args:
        :param model_dir:
        :param model:
        """
        if controller == cls.OMNISCIENT_CONTROLLER:
            controller_factory = controllers_task1.OmniscientController
        elif re.match(cls.LEARNED_CONTROLLER, controller):
            net = torch.load('%s/%s' % (model_dir, model))

            def controller_factory():
                return controllers_task1.LearnedController(net=net)
        else:
            raise ValueError("Invalid value for controller")

        world, marxbot, d_object = cls.setup(controller_factory)

        # Generate random polar coordinates to define the area in which the
        # marXbot can spawn, in particular theta ∈ [0, 2π] and r ∈ [d_object.radius * 2, max_range * 1.2]
        max_range = 150  # corresponds to the proximity sensors maximal range

        # Compensate for the higher density of points at smaller values of r. This
        # is accomplished by uniformly sampling the square of r.
        # Source: https://stats.stackexchange.com/a/120535
        rmin, rmax = np.array([d_object.radius * 2, max_range * 1.2]) ** 2
        r = np.sqrt(np.random.uniform(rmin, rmax, n_simulations))

        # The angle is chosen randomly in all its possible realisations
        theta = np.random.uniform(0, 2 * np.pi, n_simulations)
        angle = np.random.uniform(0, 2 * np.pi, n_simulations)

        marxbot_rel_poses = np.array([r, theta, angle]).T.reshape(-1, 3)

        builder = cls.init_dataset()
        for n in tqdm(range(n_simulations)):
            try:
                template = builder.create_template(run=n)

                cls.init_positions(marxbot, d_object, marxbot_rel_poses[n])
                cls.run(marxbot, world, builder, template, args.gui)
            except Exception as e:
                print('ERROR: ', e)

        dataset = builder.finalize()

        return dataset

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
    def init_dataset(cls):
        """

        :return:
        """
        return DatasetBuilder({
            "name": (),
            "initial_position": ("axis"),
            "initial_angle": (),
            "goal_position": ("axis"),
            "goal_angle": (),

            "position": ("axis"),
            "angle": (),
            "wheel_target_speeds": ("wheel"),
            "scanner_distances": ("scanner_angle"),
            "scanner_image": ("scanner_angle", "channel"),

            "goal_reached": (),
            "goal_position_distance": (),
            "goal_angle_distance": ()

        }, coords={
            # TODO: run and step might be converted to a MultiIndex, making it
            #       possible to directly use them for indexing
            "run": (),
            "step": (),

            "axis": (..., ["x", "y"]),
            "channel": (..., ["r", "g", "b"]),
            "wheel": (..., ["l", "r"]),
            "scanner_angle": (..., np.linspace(-np.pi, np.pi, 180))
        })

    @classmethod
    def update_template_robot(cls, template: DatasetBuilder.Template, marxbot):
        """

        :param template:
        :param marxbot:
        :return:
        """
        template.update(
            name=marxbot.name,
            initial_position=marxbot.initial_position,
            initial_angle=marxbot.initial_angle,
            goal_position=marxbot.goal_position,
            goal_angle=marxbot.goal_angle
        )

    @classmethod
    def update_template_step(cls, template: DatasetBuilder.Template, marxbot, step):
        """

        :param template:
        :param marxbot:
        :param step:
        :return:
        """
        template.update(
            step=step,
            position=marxbot.position,
            angle=marxbot.angle,
            wheel_target_speeds=marxbot.wheel_target_speeds,
            scanner_distances=np.array(marxbot.scanner_distances),
            scanner_image=np.array(marxbot.scanner_image),
            goal_reached=marxbot.goal_reached,
            goal_position_distance=euclidean_distance(marxbot.goal_position, marxbot.position),
            goal_angle_distance=abs(angle_difference(marxbot.goal_angle, marxbot.angle))
        )

    @classmethod
    def run(cls, marxbot, world, builder, template, gui=False, T=20, dt=0.1):
        """
        Run the simulation as fast as possible or using the real time GUI.
        :param marxbot: MyMarxbot
        :param world: pyenki.World
        :param builder: DatasetBuilder
        :param template: DatasetBuilder.Template
        :param gui: bool
        :param T: float
        :param dt: float update timestep in seconds, should be below 1 (typically .02-.1)
        """

        if gui:
            # We can either run a simulation [in real-time] inside a Qt application
            world.run_in_viewer(cam_position=(100, 0), cam_altitude=350.0, orthographic=True, period=0.1)
        else:
            steps = int(T // dt)

            cls.update_template_robot(template, marxbot)

            for s in range(steps):
                world.step(dt)

                cls.update_template_step(template, marxbot, step=s)
                builder.append_sample(template)

                # Check if the robot has reached the target
                if marxbot.goal_reached:
                    break
