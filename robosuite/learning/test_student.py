if __name__ == "__main__":
    import robosuite as suite
    from robosuite.utils.input_utils import *
    from robosuite.controllers import load_controller_config
    from robosuite.wrappers import GymWrapper
    import mujoco
    import time
    import argparse
    import torch
    from model import ActorCriticNet
    from student_utils import Storage, MULTITCN, TCN
    import random
    import pandas as pd

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--policy", help="policy network name", default=None)
    parser.add_argument("-s", "--student", help="student network name", default=None)
    parser.add_argument("--save", default=False)
    parser.add_argument("--save_state", default=False)
    parser.add_argument("--param1", default=10)
    parser.add_argument("--param2", default=5)
    parser.add_argument("-n", "--name", help="experiment name", default=None)
    parser.add_argument("--render", default=False)
    args = parser.parse_args()
    experiment_name = args.name
    save_dir = "student_progress/{}/".format(args.name)
    #save_dir = "student_progress/02-24-analysis/"

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
            param1=args.param1,
            param2=args.param2,
        )
    )
    env.set_curriculum_param(1.0)
    obs, _ = env.reset()

    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]

    model = ActorCriticNet(num_inputs, num_outputs, [256, 256])
    model.load_state_dict(
        torch.load(
            args.policy,
            map_location=torch.device(device),
        )
    )
    model.to(device)
    model.set_noise(-2.5 * np.ones(num_outputs))

    horizon = 20
    sensor_dim = 6
    privileged_dim = 10
    storage = Storage(sensor_dim=sensor_dim, privileged_dim=privileged_dim, horizon=horizon, max_size=1000)

    student_encoder = MULTITCN(input_size=sensor_dim, output_size=privileged_dim, num_channels=[256, 256], kernel_size=5).to(device)
    student_encoder.load_state_dict(
        torch.load(
            args.student,
            map_location=torch.device(device),
        )
    )

    sleep_time = 1.0 / env.control_freq

    tic = time.time()
    success_count = []
    sum_success_count = []
    max_timestep = []
    X=[]
    Y=[]
    A=[]
    sum_success = 0
    episode = 0
    max_episode = 1
    for i in range(10000000):
        success = 0
        with torch.no_grad():
            if storage.get_size() > 0:
                sensor_history = storage.get_recent_sensor_history()[None, :]
                sensor_history_swap = torch.swapaxes(sensor_history, 1, 2)
                latent = student_encoder(sensor_history_swap)[None, :]

                obs_student = torch.from_numpy(obs.astype("float32")[None, :]).to(device)
                obs_student[:, sensor_dim:] = latent[0, :, :].clone()

                act = model.sample_best_actions(obs_student).cpu().numpy()[0]
            else:
                obs_student = torch.from_numpy(obs.astype("float32")[None, :]).to(device)
                act = model.sample_best_actions(obs_student).cpu().numpy()[0]

        command = act

        obs, rew, terminated, truncated, info = env.step(command)

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
            # assert rew >= 1.0  # make sure that the success bonus was applied
            truncated = True
            terminated = False
            success = 1

        sum_success += success
        # wrist_pos_rel = obs[0:3]
        # wrist_force = obs[3:6]
        # peg_pos_rel = obs[6:9]
        # peg_rot6d = obs[9:15]

        done = terminated or truncated

        sensor = obs[:sensor_dim].copy()
        privileged = obs[sensor_dim:].copy()
        action = command.copy()
        sensor_torch = torch.from_numpy(sensor.astype("float32")[None, :]).to(device)
        action_torch = torch.from_numpy(action.astype("float32")[None, :]).to(device)            
        concat_torch = torch.cat((sensor_torch, action_torch),dim=1)
        privileged_torch = torch.from_numpy(privileged.astype("float32")[None, :]).to(device)
        storage.push(sensor_torch, privileged_torch, done)
        if i>0:
            X.append(np.append(obs.copy(),latent.cpu().numpy()))
        toc = time.time() - tic
        if args.render:
            env.render()
            if  obs_student[:, -1] >= 0.99:
                env.sim.model.geom("gripper0_peg_visual").rgba = [0.7, 0.0, 0.0, 1.0]
        if done:
            print("success & total success", success, sum_success)
            success_count.append(success)
            max_timestep.append(env.timestep)
            sum_success_count.append(sum_success)
            data = {
            "success": success_count,
            "sum success": sum_success_count,
            "termination step": max_timestep,
            }
            data_state = X
            obs, info = env.reset()
            env.sim.model.geom("gripper0_peg_visual").rgba = [0.0, 0.7, 0.7, 1.0]
            storage.clear()
            if args.save:
                df = pd.DataFrame(data)
                df.to_csv(save_dir + "{}_success_rates.csv".format(experiment_name), index=False)
            if args.save_state:
                df = pd.DataFrame(data_state)
                df.to_csv(save_dir + "{}_states.csv".format(experiment_name), index=False)
            episode += 1
            # env.close()
        
            # print(time.time() - tic)

        if episode >= max_episode:
            break
        tic = time.time()
