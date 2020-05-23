import re

import numpy as np
import pyenki
from tqdm import tqdm

from controllers import controllers_task1
from dataset import DatasetBuilder
from geometry import Point, Transform
from kinematics import euclidean_distance, angle_difference
from marxbot import MyMarxbot
from neural_networks import load_network


class GenerateSimulationData:
    OMNISCIENT_CONTROLLER = "omniscient"
    LEARNED_CONTROLLER = r"^learned"

    @classmethod
    def generate_initial_poses(cls, mode, n_simulations):
        if mode == 'uniform':
            # Generate random polar coordinates to define the area in which the
            # marXbot can spawn, in particular theta ∈ [0, 2π] and r ∈ [0, max_range * 1.2]
            max_range = 150  # corresponds to the proximity sensors maximal range

            # Compensate for the higher density of points at smaller values of r. This
            # is accomplished by uniformly sampling the square of r.
            # Source: https://stats.stackexchange.com/a/120535
            rmin, rmax = np.array([0, max_range * 1.2]) ** 2
            r = np.sqrt(np.random.uniform(rmin, rmax, n_simulations))

            # The angle is chosen randomly in all its possible realisations
            theta = np.random.uniform(0, 2 * np.pi, n_simulations)
            angle = np.random.uniform(0, 2 * np.pi, n_simulations)

        elif mode == 'demo':
            assert n_simulations == 7, "Demo mode only supports a fixed number of simulations"

            origin_r = 38.98

            r     = [origin_r,    origin_r,  2 * origin_r, 2 * origin_r,  2 * origin_r,           200.0,           200.0]
            theta = [   np.pi, -np.pi / 12, 3 * np.pi / 4,        np.pi, 5 * np.pi / 4,      -np.pi / 4,   7 * np.pi / 8]
            angle = [     0.0,  np.pi /  2, 5 * np.pi / 4,          0.0,     np.pi / 2,      -np.pi / 6,      -np.pi / 6]
        else:
            raise ValueError("Unknown initial_poses mode '%s'" % mode)

        initial_poses = np.array([r, theta, angle]).T.reshape(-1, 3)

        return initial_poses

    @classmethod
    def save_initial_poses(cls, file, initial_poses):
        np.save(file, initial_poses)

    @classmethod
    def load_initial_poses(cls, file):
        return np.load(file)

    @classmethod
    def generate_simulation(cls, n_simulations, controller, goal_object, model_dir, initial_poses, gui=False):
        """

        :param n_simulations:
        :param controller:
        :param goal_object:
        :param gui:
        :param model_dir:
        :param initial_poses:
        """
        if controller == cls.OMNISCIENT_CONTROLLER:
            controller_factory = controllers_task1.OmniscientController
        elif re.match(cls.LEARNED_CONTROLLER, controller):
            net = load_network(model_dir)

            def controller_factory():
                return controllers_task1.LearnedController(net=net)
        else:
            raise ValueError("Invalid value for controller")

        world, marxbot, d_object = cls.setup(controller_factory, goal_object)

        builder = cls.init_dataset(goal_object)
        for n in tqdm(range(n_simulations)):
            try:
                template = builder.create_template(run=n)

                cls.init_positions(marxbot, d_object, initial_poses[n])
                cls.run(marxbot, world, builder, template, gui)
            except Exception as e:
                print('ERROR: ', e)
                import traceback
                traceback.print_tb(e.__traceback__)

        dataset = builder.finalize()

        return dataset

    @classmethod
    def setup_docking_station(cls, size, height, goal_object):

        if goal_object == 'coloured_station':
            face_colours = [pyenki.Color(0.839, 0.153, 0.157),
                            pyenki.Color(1.000, 0.895, 0.201),
                            pyenki.Color(0.173, 0.627, 0.173)]
        elif goal_object == 'station':
            face_colours = [pyenki.Color(0, 0.5, 0.5), pyenki.Color(0, 0.5, 0.5), pyenki.Color(0, 0.5, 0.5)]
        else:
            raise ValueError("Invalid value for goal_object")

        d_object = pyenki.CompositeObject(
            [([(0, 1 * size), (0, 0.5 * size), (2 * size, 0.5 * size), (2 * size, 1 * size)], height,
              [face_colours[0]] * 4),
             ([(0, -0.5 * size), (0, -1 * size), (2 * size, -1 * size), (2 * size, -0.5 * size)], height,
              [face_colours[2]] * 4),
             ([(0, 0.5 * size), (0, -0.5 * size), (0.5 * size, -0.5 * size), (0.5 * size, 0.5 * size)], height,
              [face_colours[1]] * 4)],
            -1, pyenki.Color(0, 0.5, 0.5))

        return d_object

    @classmethod
    def setup(cls, controller_factory, goal_object):
        """
        Set up the world.
        :param controller_factory: if the controller is passed, load the learned network
        :param goal_object: influence the colour of the docking station
        :return world, marxbot, docking_station

        """

        # Create an unbounded world
        world = pyenki.World()

        marxbot = MyMarxbot(name='marxbot%d' % 1, controller=controller_factory())
        world.add_object(marxbot)

        size = 20.0
        height = 40.0

        d_object = cls.setup_docking_station(size, height, goal_object)
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
    def init_dataset(cls, goal_object):
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
        }, attrs={
            "goal_object": goal_object
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
