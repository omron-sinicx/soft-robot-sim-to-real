from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import (
    GuriguriLargeSquareHoleObject,
    GuriguriLargeRoundHoleObject,
    GuriguriLargeRectangleHoleObject,
    GuriguriLargeTriangleHoleObject,
)
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import *
from robosuite.controllers import controller_factory


class SoftPegInHole(SingleArmEnv):
    """
    This class corresponds to the peg in hole task for a single arm
    It is mostly copied and pasted from SoftPegInHole from Hai's robotsuite repo (which was then mostly based
    off of Lift)
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise={"magnitude": 0.002, "type": "gaussian"},
        table_full_size=(0.65, 0.65, 0.025),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=False,
        reward_scale=1.0,
        reward_shaping=True,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=200,
        ignore_done=False,
        hard_reset=False,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        deterministic_reset=False,
        param1=0.0,
        param2=0.0,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0.0, 0.65, 0.025))
        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self.total_rewards = 0.0

        self.set_curriculum_param(0.0)
        self.param1 = float(param1)
        self.param2 = float(param2)

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types=None,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

        # override base.py which hard codes it to be false, such that it can be set by the user
        # TODO: make it also toggle hole placement randomization
        self.deterministic_reset = deterministic_reset

    def reward(self, action=None):
        """
        Reward function for the task.
        """
        reward = 0.0

        # note: I don't bother making use of features like reward_shaping, reward_scale or _check_success

        peg_pos = self.sim.data.get_site_xpos("gripper0_peg_ft_frame").copy()
        hole_pos = self.sim.data.body_xpos[self.hole_body_id].copy()
        hole_pos[2] -= 0.015
        peg_error = peg_pos - hole_pos
        weights = np.array([1.0, 1.0, 10.0])
        weighted_peg_dist = np.linalg.norm(np.sqrt(weights) * peg_error)
        # peg_reward = np.dot(peg_error, weights * peg_error)

        # peg_quat_target = 1.0 / np.sqrt(2) * np.array([-1.0, -1.0, 0.0, 0.0])
        # peg_quat = mat2quat(self.sim.data.get_site_xmat("gripper0_peg_ft_frame"))
        # peg_quat_error = np.linalg.norm(quat_distance(peg_quat, peg_quat_target)[:3])
        # # peg_quat_error == 0.01 roughly correponds to +-1 degree error

        # if np.linalg.norm(peg_error[:2]) < 0.007 and peg_quat_error < self.param1:
        if np.linalg.norm(peg_error[:2]) < 0.007:
            action_reward = 0.001 * action[2] ** 2.0
        else:
            action_reward = 1.0 * action[2] ** 2.0

        action_smoothness_reward = 1.0 * np.linalg.norm(action - self.action_prev) ** 2.0

        # if self._check_success() == True:
        #     reward += 1.0 + self.horizon - self.timestep
        # else:
        #     reward += 1.0 - 10.0 * (peg_reward + action_reward)

        reward += (
            1.0 * (self.weighted_peg_dist_prev - weighted_peg_dist) / 0.001 - action_reward - action_smoothness_reward
        )

        self.total_rewards += reward

        self.weighted_peg_dist_prev = weighted_peg_dist

        return reward

    def _post_action(self, action):
        """
        Additional termination conditions compared to definition given in base.py
        Args:
            action (np.array): Action to execute within the environment
        Returns:
            3-tuple:
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) empty dict to be filled with information by subclassed method
        """
        reward, self.done, _ = super()._post_action(action)

        # additional termination conditions compared to super
        if self.ignore_done == True:
            self.done = False
        else:
            # kinmatic singularity termination
            self.done = self.done or np.linalg.det(self.robots[0].controller.J_full) < 0.01

            # TODO: I have this  code repeated in many locations. I really need to do some refactoring...
            peg_pos = self.sim.data.get_site_xpos("gripper0_peg_ft_frame").copy()
            hole_pos = self.sim.data.body_xpos[self.hole_body_id].copy()
            hole_pos[2] -= 0.015
            peg_error = peg_pos - hole_pos
            weights = np.array([1.0, 1.0, 10.0])
            weighted_peg_dist = np.linalg.norm(np.sqrt(weights) * peg_error)

            self.done = self.done or weighted_peg_dist > self.weighted_peg_dist_init * 1.2

        self.action_prev = action.copy()

        if self._check_success():
            reward += 1.0

        if self.done:
            reward += -5.0

        return reward, self.done, {}

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        self.hole = GuriguriLargeRoundHoleObject(name="hole")
        # self.hole = GuriguriLargeSquareHoleObject(name="hole")
        # self.hole = GuriguriLargeTriangleHoleObject(name="hole")
        # self.hole = GuriguriLargeRectangleHoleObject(name="hole")

        # TODO: toggle placement initialization randomization based on self.deterministic_reset
        # defalt hole range = 0.005
        hole_range = self.param1*0.001

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.hole)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.hole,
                x_range=[-hole_range, hole_range],
                y_range=[-hole_range, hole_range],
                # TODO: randomize rotation as well, but this I need to take observations relative to hole quaternions then
                rotation=0.0,
                # rotation=np.pi / 2.0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.0,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            # mujoco_objects=[self.box, self.hole],
            mujoco_objects=[self.hole],
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.hole_body_id = self.sim.model.body_name2id(self.hole.root_body)
        # self.box_body_id = self.sim.model.body_name2id(self.box.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """

        # overwrite parent setup observables call
        # the code below mirrors the code in robot/robots.py

        # Get prefix from robot model to avoid naming clashes for multiple robots and define observables modality
        pf = self.robots[0].robot_model.naming_prefix
        modality = f"{pf}proprio"

        @sensor(modality=modality)
        def wrist_pos_rel(obs_cache):
            wrist_pos = self.sim.data.get_body_xpos("gripper0_gripper_base")
            # hole_pos = self.sim.data.body_xpos[self.hole_body_id].copy()
            # hole_pos[2] -= 0.015
            hole_pos_nominal = np.array([0.0, 0.65, 0.15])

            offset = np.array([0.0, 0.0, -0.23])
            scale = 0.1
            # return (wrist_pos - hole_pos + offset) / scale]
            return (wrist_pos - hole_pos_nominal + offset) / scale

        @sensor(modality=modality)
        def wrist_rot6d(obs_cache):
            return self.sim.data.get_body_xmat("gripper0_gripper_base")[:, :2].flatten()

        @sensor(modality=modality)
        def wrist_wrench(obs_cache):
            # NOTE self.sim.data.get_sensor doesn't seem to obtain the full sensor vector
            wrist_wrench = np.hstack(
                (
                    self.sim.data._data.sensor("gripper0_force_ee").data,
                    self.sim.data._data.sensor("gripper0_torque_ee").data,
                )
            )
            return wrist_wrench

        @sensor(modality=modality)
        def wrist_force(obs_cache):
            # NOTE self.sim.data.get_sensor doesn't seem to obtain the full sensor vector
            wrist_force = self.sim.data._data.sensor("gripper0_force_ee").data
            offset = np.array([0.0, 0.0, 10.0])
            scale = 10.0
            return (wrist_force + offset) / scale

        @sensor(modality=modality)
        def spring_angle(obs_cache):
            return np.array(
                [
                    self.sim.data.get_joint_qpos("gripper0_flex_wrist_rx"),
                    self.sim.data.get_joint_qpos("gripper0_flex_wrist_ry"),
                    self.sim.data.get_joint_qpos("gripper0_flex_wrist_rz"),
                ]
            )

        @sensor(modality=modality)
        def peg_torque(obs_cache):
            return self.sim.data._data.sensor("gripper0_torque_peg").data

        @sensor(modality=modality)
        def peg_pos_rel(obs_cache):
            peg_pos = self.sim.data.get_site_xpos("gripper0_peg_ft_frame")
            hole_pos = self.sim.data.body_xpos[self.hole_body_id].copy()
            hole_pos[2] -= 0.015
            # hole_pos_nominal = np.array([0.0, 0.65, 0.15])

            # no offset because hole pos is defined precisely to match peg pos after successful insertion
            scale = 0.1
            # return (peg_pos - hole_pos_nominal) / scale
            return (peg_pos - hole_pos) / scale

        @sensor(modality=modality)
        def peg_rot6d(obs_cache):
            return self.sim.data.get_site_xmat("gripper0_peg_ft_frame")[:, :2].flatten()

        @sensor(modality=modality)
        def joint_gains(obs_cache):
            # assume that all gains are common across joints
            assert np.linalg.norm(np.diff(self.robots[0].controller.kp)) < 10e-6
            assert np.linalg.norm(np.diff(self.robots[0].controller.kd)) < 10e-6

            raw_gains = np.array([self.robots[0].controller.kp[0], self.robots[0].controller.kd[0]])
            scale = np.array([5.0, 3.0])
            return np.log10(raw_gains) / scale

        @sensor(modality=modality)
        def hole_offset(obs_cache):
            hole_pos = self.sim.data.body_xpos[self.hole_body_id].copy()
            hole_pos[2] -= 0.015

            hole_pos_nominal = np.array([0.0, 0.65, 0.15])
            scale = 0.01
            return (hole_pos - hole_pos_nominal) / scale

        @sensor(modality=modality)
        def peg_angle(obs_cache):
            return self.peg_angle / (5.0 * np.pi / 180.0)

        @sensor(modality=modality)
        def peg_alignment(obs_cache):
            peg_pos = self.sim.data.get_site_xpos("gripper0_peg_ft_frame").copy()
            hole_pos = self.sim.data.body_xpos[self.hole_body_id].copy()
            hole_pos[2] -= 0.015
            peg_error = peg_pos - hole_pos

            # peg_quat_target = 1.0 / np.sqrt(2) * np.array([-1.0, -1.0, 0.0, 0.0])
            # peg_quat = mat2quat(self.sim.data.get_site_xmat("gripper0_peg_ft_frame"))
            # peg_quat_error = np.linalg.norm(quat_distance(peg_quat, peg_quat_target)[:3])
            # # peg_quat_error == 0.01 roughly correponds to +-1 degree error

            # aligned = np.linalg.norm(peg_error[:2]) < 0.007 and peg_quat_error < self.param1
            aligned = np.linalg.norm(peg_error[:2]) < 0.007
            return aligned

        # @sensor(modality=modality)
        # def peg_vel(obs_cache):
        #     return self.sim.data.get_site_xvelp("gripper0_peg_ft_frame")

        # @sensor(modality=modality)
        # def wrist_vel(obs_cache):
        #     return self.sim.data.get_body_xvelp("gripper0_gripper_base")

        # @sensor(modality=modality)
        # def peg_omega(obs_cache):
        #     return self.sim.data.get_site_xvelr("gripper0_peg_ft_frame")

        # @sensor(modality=modality)
        # def wrist_omega(obs_cache):
        #     return self.sim.data.get_body_xvelr("gripper0_gripper_base")

        # # time phase
        # # this isn't really proprioception, but whatever...
        # @sensor(modality=modality)
        # def time_phase(obs_cache):
        #     return np.array(
        #         [
        #             np.cos(2.0 * np.pi * self.timestep / self.horizon),
        #             np.sin(2.0 * np.pi * self.timestep / self.horizon),
        #         ]
        #     )

        sensors = [
            wrist_pos_rel,
            # wrist_rot6d,
            # wrist_wrench,
            wrist_force,
            # peg_torque,
            # spring_angle,
            peg_pos_rel,
            peg_rot6d,
            # peg_vel,
            # wrist_vel,
            # peg_omega,
            # wrist_omega,
            # time_phase,
            # joint_gains,
            # hole_offset,
            # peg_angle,
            peg_alignment,
        ]
        names = [
            "wrist_pos_rel",
            # "wrist_rot6d",
            # "wrist_wrench",
            "wrist_force",
            # "peg_torque",
            # "spring_angle",
            # "peg_pos_rel",
            "peg_rot6d",
            "peg_vel",
            # "wrist_vel",
            # "peg_omega",
            # "wrist_omega",
            # "time_phase",
            # "joint_gains",
            # "hole_offset",
            # "peg_angle",
            "peg_alignment",
        ]
        actives = [True] * len(sensors)

        # Create observables for this robot
        observables = OrderedDict()
        for name, s, active in zip(names, sensors, actives):
            obs_name = pf + name
            observables[obs_name] = Observable(
                name=obs_name,
                sensor=s,
                sampling_rate=self.control_freq,
                active=active,
            )

        return observables

    def set_curriculum_param(self, new_curriculum_paramter):
        self.curriculum_param = new_curriculum_paramter

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler
        # if not self.deterministic_reset:
        # Sample from the placement initializer for all objects
        object_placements = self.placement_initializer.sample()

        # Loop through all objects and reset their positions
        for obj_pos, obj_quat, obj in object_placements.values():
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # y direction offset position
        new_init_qpos = np.array(
            [
                1.4247818792966438,
                -1.2206764009655182,
                1.3337569807166205,
                -1.683851374789445,
                -1.5707984375494854,
                -1.7167519113643595,
            ]
        )

        self.robots[0].set_init_qpos(new_init_qpos)
        self.robots[0].controller.update_initial_joints(new_init_qpos)
        self.robots[0].controller.reset_goal()

        self.robots[0].controller.kp = np.ones(6) * 10.0 ** np.random.uniform(3.0, 4.0)
        self.robots[0].controller.kd = 2 * np.sqrt(self.robots[0].controller.kp) * np.random.uniform(0.1, 1.0)

        # if np.random.uniform(0.0, 1.0) < self.param2:
        # default angle 5 deg
        if True:
            self.peg_angle = self.param2 * np.random.uniform(-1.0, 1.0) * np.pi / 180.0
        else:
            self.peg_angle = 0.0
        rot_quat = axisangle2quat(np.array([0.0, self.peg_angle, 0.0]))

        # TODO: rotate visual peg 90 degrees in z direction
        self.sim.model._model.geom("gripper0_peg_visual").quat = quat_multiply(np.array([0.0, 1.0, 0.0, 0.0]), rot_quat)
        self.sim.model._model.geom("gripper0_peg_collision").quat = quat_multiply(
            np.array([0.0, 1.0, 0.0, 0.0]), rot_quat
        )
        self.sim.model._model.site("gripper0_peg_ft_frame").quat = quat_multiply(
            np.array([1.0, 0.0, 0.0, 0.0]), rot_quat
        )

        self.peg_pos_init = self.sim.data.get_site_xpos("gripper0_peg_ft_frame").copy()

        peg_pos = self.sim.data.get_site_xpos("gripper0_peg_ft_frame").copy()
        hole_pos = self.sim.data.body_xpos[self.hole_body_id].copy()
        hole_pos[2] -= 0.015
        peg_error = peg_pos - hole_pos
        weights = np.array([1.0, 1.0, 10.0])
        weighted_peg_dist = np.linalg.norm(np.sqrt(weights) * peg_error)

        self.action_prev = np.zeros(self.action_dim)
        self.peg_pos_init = peg_pos
        self.weighted_peg_dist_prev = weighted_peg_dist
        self.weighted_peg_dist_init = weighted_peg_dist

        self.total_rewards = 0.0

        # # Reset all object positions using initializer sampler if we're not directly loading from an xml
        # if not self.deterministic_reset:

        #     # Sample from the placement initializer for all objects
        #     object_placements = self.placement_initializer.sample()

        #     # Loop through all objects and reset their positions
        #     for obj_pos, obj_quat, obj in object_placements.values():
        #         # temporary hack
        #         if len(obj.joints) > 0:
        #             self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        TODO
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # TODO: additional visualization

    def _check_success(self):
        peg_pos = self.sim.data.get_site_xpos("gripper0_peg_ft_frame").copy()
        hole_pos = self.sim.data.body_xpos[self.hole_body_id].copy()
        hole_pos[2] -= 0.015
        peg_error = peg_pos - hole_pos

        if np.linalg.norm(peg_error) < 0.005:
            return True
        else:
            return False        
