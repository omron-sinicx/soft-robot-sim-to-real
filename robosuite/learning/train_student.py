if __name__ == "__main__":
    import robosuite as suite
    from robosuite.utils.input_utils import *
    from robosuite.controllers import load_controller_config
    from robosuite.wrappers import GymWrapper
    import mujoco
    import time
    import argparse
    import random
    import torch
    import torch.nn as nn
    from model import ActorCriticNet
    import matplotlib.pyplot as plt
    from datetime import datetime
    import os
    from student_utils import Storage, TCN, MULTITCN

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        nargs="?",
        help='.pt file name. no need to include "learning_progress/"',
    )
    parser.add_argument("--param1", default=10)
    parser.add_argument("--param2", default=5)
    parser.add_argument("-n", "--name", help="experiment name", default=None)
    args = parser.parse_args()
    if args.name is None:
        args.name = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    save_dir = "student_progress/{}/".format(args.name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    seed = 1
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
            gripper_types="RobotiqHandEGripperSoft",
            # has_renderer=True,
            controller_configs=controller_configs,
            # ignore_done=True,
            use_camera_obs=False,
            # deterministic_reset=True,
            param1=args.param1,
            param2=args.param2,
        )
    )
    print("env_created")
    env.set_curriculum_param(1.0)
    obs, _ = env.reset()

    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]

    assert args.filename is not None
    model = ActorCriticNet(num_inputs, num_outputs, [256, 256])
    model.load_state_dict(
        torch.load(
            args.filename,
            map_location=torch.device(device),
        )
    )
    model.to(device)
    model.set_noise(-2.5 * np.ones(num_outputs))

    sleep_time = 1.0 / env.control_freq

    losses = []
    storage_size = 1000
    minibatch_size = storage_size // 4
    horizon = 20
    sensor_dim = 6
    privileged_dim = 10
    policy_brend = 0.999
    storage = Storage(sensor_dim=sensor_dim, privileged_dim=privileged_dim, horizon=horizon, max_size=storage_size)

    student_encoder = MULTITCN(input_size=sensor_dim, output_size=privileged_dim, num_channels=[256, 256], kernel_size=5).to(device)
    loss_function = nn.MSELoss()
    loss_function_align = nn.BCELoss()
    optimizer = torch.optim.Adam(student_encoder.parameters(), lr=3e-4)

    tic = time.time()
    for iter in range(10000):
        while storage.get_size() < storage_size:
            with torch.no_grad():
                if storage.get_size() > 0:
                    student_encoder.eval()
                    sensor_history = storage.get_recent_sensor_history()[None, :]
                    sensor_history_swap = torch.swapaxes(sensor_history, 1, 2)
                    latent = student_encoder(sensor_history_swap)[None, :]

                    obs_student = torch.from_numpy(obs.astype("float32")[None, :]).to(device)
                    obs_student[:, sensor_dim:] = latent[0, :, :].clone()

                    act = model.sample_best_actions(obs_student).cpu().numpy()[0]
                else:
                    obs = torch.from_numpy(obs.astype("float32")[None, :]).to(device)
                    act = model.sample_best_actions(obs).cpu().numpy()[0]

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

            # wrist_pos_rel = obs[0:3]
            # wrist_force = obs[3:6]
            # peg_pos_rel = obs[6:9]
            # peg_rot6d = obs[9:15]

            # wrist_rot6d_reshape = wrist_rot6d.reshape((3, 2))
            # peg_rot6d_reshape = peg_rot6d.reshape((3, 2))

            done = terminated or truncated

            sensor = obs[:sensor_dim].copy()
            privileged = obs[sensor_dim:].copy()
            sensor_torch = torch.from_numpy(sensor.astype("float32")[None, :]).to(device)
            privileged_torch = torch.from_numpy(privileged.astype("float32")[None, :]).to(device)
            storage.push(sensor_torch, privileged_torch, done)

            if done:
                obs, info = env.reset()
                # env.close()

            # env.render()

        student_encoder.train()
        sensor_history_minibatch, privileged_minibatch = storage.sample(minibatch_size=minibatch_size)
        optimizer.zero_grad()
        sensor_history_minibatch_swap = torch.swapaxes(sensor_history_minibatch, 1, 2)
        student_encoder_output = student_encoder(sensor_history_minibatch_swap) 
        loss_pose = loss_function(student_encoder_output[:,:-1], privileged_minibatch[:,:-1])
        loss_align = loss_function_align(student_encoder_output[:,-1:], privileged_minibatch[:,-1:])
        loss = loss_pose + 0.1*loss_align
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        storage.clear()

        if iter % 10 == 0:
            plt.clf()
            plt.plot(np.log(losses), color="b")
            plt.grid("True")
            plt.savefig(save_dir + "latent_loss.png")
            torch.save(student_encoder.state_dict(), save_dir + "latest.pt".format(iter))

        if iter % 1000 == 0:
            torch.save(student_encoder.state_dict(), save_dir + "iter{}.pt".format(iter))

        print(iter)
