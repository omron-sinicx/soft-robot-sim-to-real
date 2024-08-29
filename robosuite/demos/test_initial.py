# the second script written to test simulation, open loop control of joints

import robosuite as suite
from robosuite.utils.input_utils import *
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper
import mujoco
import time

if __name__ == "__main__":
    action_dim = 6
    control_freq = 20

    controller_configs = load_controller_config(default_controller="OSC_POSITION")
    controller_configs["control_delta"] = False

    # initialize the task
    env = GymWrapper(
        suite.make(
            env_name="SoftPegInHole",
            robots="UR5e",
            gripper_types=None,
            # gripper_types="RobotiqHandEGripper",
            # gripper_types="RobotiqHandEGripperSoft",
            controller_configs=controller_configs,
            # has_renderer=True,
            has_offscreen_renderer=False,
            hard_reset=False,
            ignore_done=True,
            use_camera_obs=False,
            horizon=1000,
            control_freq=control_freq,
        )
    )
    obs, _ = env.reset()
    # env.viewer.set_camera(camera_id=3)
    # env.sim._render_context_offscreen.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

    # Define neutral value
    assert len(env.robots) == 1
    gripper_dim = env.robots[0].gripper.dof
    # neutral = np.zeros(action_dim + gripper_dim)
    neutral = np.array([-0.222, -0.0294,  1.13])

    tic = time.time()
    for i in range(10000):
        sec = i * 1.0 / control_freq

        action = neutral.copy()
        # action_idx = int(sec) % 6  # 6 DOF action
        # action_sign = 2.0 * (int(sec) % 12 < 5) - 1  # alternate directions
        # action_value = (
        #     0.49 * (int(sec) % 6 > 2) + 0.01
        # )  # larger values for radian angles
        # action[action_idx] = action_sign * action_value

        # action[3] = 0.2 * np.sin(sec)
        # action[-1] = -1.0 + 2.0 * (int(sec) % 2 == 0)

        # action[2] += 0.05 * np.sin(sec)

        obs, rew, terminated, truncated, info = env.step(action)
        # env.render()
        # import ipdb; ipdb.set_trace()

        # from scipy.spatial.transform import Rotation
        # ori = Rotation.from_quat(obs[3:])
        # rot_180_x = Rotation.from_quat([0.0, 1.0, 0.0, 0.0])
        # print((rot_180_x * ori).as_euler("xyz"))

        if terminated or truncated or i % 30 == 0:
            obs, info = env.reset()

        # # sleep to make animation realtime
        # toc = time.time() - tic
        # time.sleep(max(1.0/control_freq - toc, 0))
        # tic = time.time()

    env.close()
