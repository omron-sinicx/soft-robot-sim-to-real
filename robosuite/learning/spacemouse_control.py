if __name__ == "__main__":
    import robosuite as suite
    from robosuite.utils.input_utils import *
    from robosuite.controllers import load_controller_config
    from robosuite.wrappers import GymWrapper
    import mujoco
    import time
    import argparse

    # import torch
    # from model import ActorCriticNet

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        nargs="?",
        help='.pt file name. no need to include "learning_progress/"',
    )
    args = parser.parse_args()

    # controller_configs = load_controller_config(default_controller="OSC_POSE")
    # controller_configs["control_delta"] = False

    controller_configs = load_controller_config(default_controller="JOINT_POSITION")

    env = GymWrapper(
        suite.make(
            env_name="SoftPegInHole",
            robots="UR5e",
            gripper_types=None,
            # gripper_types="RobotiqHandEGripperSoft",
            # has_renderer=True,
            controller_configs=controller_configs,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            horizon=200,
            control_freq=20,
        )
    )
    print("env_created")
    obs, _ = env.reset()

    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]

    sleep_time = 1.0 / env.control_freq

    q_ur_home = np.pi / 2.0 * np.array([-1, -1, 0, -1, 0, 0])
    q_home = np.pi / 2.0 * np.array([1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
    x_home = np.array(
        [
            -0.13075518,
            0.49268331,
            0.48719351,
            0.7084395,
            0.70576026,
            0.00297184,
            0.00266439,
        ]
    )

    # initialize device
    from robosuite.devices import Keyboard

    device = Keyboard(pos_sensitivity=0.01, rot_sensitivity=0.01)
    device.start_control()
    axis = 0

    # q_des = np.pi / 2.0 * np.array([0, -1, 0, -1, 0, 0])
    q_des = np.pi / 2.0 * np.array([1, -1, 1, -1, -1, -1])

    tic = time.time()
    for i in range(10000000):
        # command = np.zeros(num_outputs)

        keyboard_state = device.get_controller_state()["dpos"].copy()

        axis = axis + 2000 * keyboard_state[1]
        axis = int(min(max(axis, 0), 5))

        q_des[axis] += 10.0 * keyboard_state[0]

        command = 10.0 * (q_des - env.robots[0]._joint_positions.copy())

        # print(np.hstack((env.robots[0]._hand_pos, env.robots[0]._hand_quat)))

        obs, rew, terminated, truncated, info = env.step(command)

        if terminated or truncated:
            obs, info = env.reset()

        # env.render()

        toc = time.time() - tic
        time.sleep(max(sleep_time - toc, 0))
        # print(time.time() - tic)
        tic = time.time()
