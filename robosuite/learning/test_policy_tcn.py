if __name__ == "__main__":
    import robosuite as suite
    from robosuite.utils.input_utils import *
    from robosuite.controllers import load_controller_config
    from robosuite.wrappers import GymWrapper
    import mujoco
    import time
    import argparse
    import torch
    import random
    import pandas as pd
    from model import ActorCriticNetTCN
    from student_utils import Storage

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        nargs="?",
        help='.pt file name. no need to include "learning_progress/"',
    )
    parser.add_argument("-n", "--name", help="experiment name", default=None)
    parser.add_argument("--save", default=False)
    parser.add_argument("--param1", default=10)
    parser.add_argument("--param2", default=5)
    parser.add_argument("--render", default=False)
    args = parser.parse_args()
    experiment_name = args.name
    save_dir = "learning_progress_tcn/{}/".format(args.name)

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    controller_configs = load_controller_config(default_controller="OSC_POSITION")

    env = GymWrapper(
        suite.make(
            env_name="SoftPegInHole",
            robots="UR5e",
            # gripper_types=None,
            # gripper_types="RobotiqHandEGripper",
            gripper_types="RobotiqHandEGripperSoft",
            has_renderer=True,
            controller_configs=controller_configs,
            # ignore_done=True,
            use_camera_obs=False,
            # deterministic_reset=True,
            param1 = args.param1,
            param2 = args.param2,
        )
    )
    print("env_created")
    env.set_curriculum_param(1.0)
    obs, _ = env.reset()

    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    horizon = 20

    if args.filename is not None:
        model = ActorCriticNetTCN(num_inputs, num_outputs, [256, 256], kernel_size=5)
        model.load_state_dict(
            torch.load(
                args.filename,
                map_location=torch.device(device),
            )
        )
        model.to(device)
        model.set_noise(-2.5 * np.ones(num_outputs))
    
    history_storage = Storage(sensor_dim=num_inputs, privileged_dim=1, horizon=horizon, max_size=1000)
    obs = torch.from_numpy(obs.astype("float32")[None, :]).to(device)
    history_storage.push(obs, torch.zeros(1).to(device), False)

    sleep_time = 1.0 / env.control_freq

    tic = time.time()
    success_count = []
    sum_success_count = []
    max_timestep = []
    sum_success = 0
    episode = 0
    max_episode = 100
    for i in range(10000000):
        success = 0
        if args.filename is not None:
            with torch.no_grad():
                inputs = torch.vstack((history_storage.get_recent_sensor_history(), obs))[None, :]
                
                act = model.sample_best_actions(inputs).cpu().numpy()[0]
                # act = model.sample_best_actions(torch.swapaxes(obs), 1, 2).cpu().numpy()[0]
        else:
            act = np.zeros(num_outputs)

        command = act

        obs, rew, terminated, truncated, info = env.step(command)
        obs = torch.from_numpy(obs.astype("float32")[None, :]).to(device)

        # override behaviour intended in robosuite (defined in base.py) such that done is
        # triggered only in termination (ie done is set to False when horizon is reached)
        # TODO: I haven't accounted for the edge case that a termination condition gets
        # triggered in the last timestep
        # note also that truncated is hard coded to false in gymwarapper
        if terminated == True and env.timestep >= env.horizon:
            truncated = True
            terminated = False

        # early terminate if insertion is successful
        if env._check_success() == True:
            # assert rew > 1.0  # make sure that the success bonus was applied
            truncated = True
            terminated = False
            success = 1

        sum_success += success
        # wrist_pos_rel = obs[0:3]
        # wrist_force = obs[3:6]
        # peg_pos_rel = obs[6:9]
        # peg_rot6d = obs[9:15]

        # wrist_rot6d_reshape = wrist_rot6d.reshape((3, 2))
        # peg_rot6d_reshape = peg_rot6d.reshape((3, 2))

        history_storage.push(obs, torch.zeros(1).to(device), terminated or truncated)

        if terminated or truncated:
            print("success & total success", success, sum_success)
            success_count.append(success)
            max_timestep.append(env.timestep)
            sum_success_count.append(sum_success)
            data = {
            "success": success_count,
            "sum success": sum_success_count,
            "termination step": max_timestep,
            }
            obs, info = env.reset()
            obs = torch.from_numpy(obs.astype("float32")[None, :]).to(device)
            env.close()
            history_storage.clear()
            history_storage.push(torch.zeros_like(obs), torch.zeros(1).to(device), False)
            if args.save:
                df = pd.DataFrame(data)
                df.to_csv(save_dir + "{}_success_rates.csv".format(experiment_name), index=False)
            episode += 1

        toc = time.time() - tic
        if args.render:
            env.render()
            time.sleep(max(sleep_time - toc, 0))
            # print(time.time() - tic)

        if episode >= max_episode:
            break
        tic = time.time()
