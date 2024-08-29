# the third script written to test simulation, control via keyboard

import robosuite as suite
from robosuite.utils.input_utils import *
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper
import mujoco
import time

if __name__ == "__main__":
    action_dim = 6
    control_freq = 20

    # initialize the task
    env = GymWrapper(
        suite.make(
            env_name="SoftPegInHole",
            robots="UR5e",
            # gripper_types=None,
            gripper_types="RobotiqHandEGripperSoft",
            controller_configs=load_controller_config(default_controller="OSC_POSITION"),
            # has_renderer=True,
            has_offscreen_renderer=False,
            hard_reset=False,
            # ignore_done=True,
            use_camera_obs=False,
            horizon=1000,
            control_freq=control_freq,
            # deterministic_reset=True,
        )
    )
    env.reset()
    # env.viewer.set_camera(camera_id=3)
    # env.sim._render_context_offscreen.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

    # initialize device
    from robosuite.devices import Keyboard

    device = Keyboard(pos_sensitivity=1.0, rot_sensitivity=1.0)
    # env.viewer.add_keypress_callback(device.on_press) #I'm not sure why keyboard control works without this...
    device.start_control()

    assert len(env.robots) == 1
    gripper_dim = env.robots[0].gripper.dof
    # neutral = np.zeros(action_dim + gripper_dim)
    active_robot = env.robots[0]

    tic = time.time()
    for i in range(10000):
        sec = i * 1.0 / control_freq

        # action = neutral.copy()

        action, grasp = input2action(device=device, robot=active_robot)

        action /= 3.75

        obs, rew, terminated, truncated, info = env.step(action)
        # env.render()

        wrist_pos_rel = obs[0:3]
        wrist_rot6d = obs[3:9]
        wrist_force = obs[9:12]
        peg_pos_rel = obs[12:15]
        peg_rot6d = obs[15:21]

        if terminated or truncated:
            print("reset")
            obs, info = env.reset()

        # # sleep to make animation realtime
        # toc = time.time() - tic
        # time.sleep(max(1.0/control_freq - toc, 0))
        # tic = time.time()

    env.close()
