import torch
import torch.nn as nn
import numpy as np
from tcn import TemporalConvNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Storage:
    def __init__(self, sensor_dim, privileged_dim, horizon, max_size=1000):
        self.max_size = max_size
        self.horizon = horizon
        self.sensor_dim = sensor_dim
        self.privileged_dim = privileged_dim

        # NOTE the zero padding
        self.sensors = torch.zeros(max_size + self.horizon, self.sensor_dim).to(device)
        self.privilegeds = torch.zeros(max_size + self.horizon, self.privileged_dim).to(device)
        self.dones = np.zeros(max_size + self.horizon, dtype=bool)
        self.dones[: self.horizon] = True

        # keep track of "size" of storage, which is also the next index to insert data in
        # NOTE that it doesn't start from zero due to the zero padding
        self.ptr = self.horizon

    def push(self, sensor, privileged, done=None):
        assert self.ptr < self.max_size + self.horizon, "tried to push to a full storage"

        self.sensors[self.ptr : self.ptr + 1] = sensor.clone()
        self.privilegeds[self.ptr : self.ptr + 1] = privileged.clone()

        if done is not None:
            self.dones[self.ptr : self.ptr + 1] = done

        self.ptr += 1

    def sample(self, minibatch_size):
        assert self.max_size >= minibatch_size

        rand_indices = np.random.choice(np.arange(self.max_size) + self.horizon, minibatch_size, replace=False)

        # NOTE the indexing: history includes sensor_{t-1},...,sensor_{t-N}, whereas privileged is privileged_{t}

        sensor_history = torch.zeros((minibatch_size, self.horizon, self.sensor_dim)).to(device)
        # TODO: make this more efficient without a loop
        for rand_idx_idx, rand_idx_value in enumerate(rand_indices):
            sensor_history[rand_idx_idx, :, :] = self.sensors[rand_idx_value - self.horizon : rand_idx_value, :].clone()

            # if done occured in local trajectory window, set all history before it to zero
            # since it's sensor data from an unrelated episode
            local_dones = self.dones[rand_idx_value - self.horizon : rand_idx_value]
            if np.any(local_dones):
                zero_mask = np.zeros_like(local_dones)
                zero_mask[: np.argwhere(local_dones)[-1, 0] + 1] = True
                sensor_history[rand_idx_idx, zero_mask, :] = 0

        return (
            sensor_history.clone(),
            self.privilegeds[rand_indices].clone(),
        )

    def get_recent_sensor_history(self):
        # TODO: handle horizon > 1
        assert self.ptr > 0, "cannot get history from empty storage"

        local_sensors = self.sensors[self.ptr - self.horizon - 1 : self.ptr - 1].clone()

        # if done occured in local trajectory window, set all history before it to zero
        # since it's sensor data from an unrelated episode
        local_dones = self.dones[self.ptr - self.horizon - 1 : self.ptr - 1].copy()
        if np.any(local_dones):
            zero_mask = np.zeros_like(local_dones)
            zero_mask[: np.argwhere(local_dones)[-1, 0] + 1] = True
            local_sensors[zero_mask, :] = 0

        return local_sensors

    def get_size(self):
        return self.ptr - self.horizon

    def clear(self):
        self.ptr = self.horizon


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return torch.tanh(o)

class MULTITCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0):
        super(MULTITCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        o_pose = torch.tanh(o[:,:-1])
        o_align = self.sigmoid(o[:,-1:])
        o_cat = torch.cat((o_pose,o_align),dim=1)
        return o_cat 


if __name__ == "__main__":
    # test Storage

    sensor_dim = 3
    privilged_dim = 2
    horizon = 5
    max_size = 20

    storage = Storage(sensor_dim=sensor_dim, privileged_dim=privilged_dim, horizon=horizon, max_size=max_size)

    for idx in range(max_size):
        sensor_torch = torch.ones((1, sensor_dim)) * (10 + idx)
        privileged_torch = torch.ones((1, privilged_dim)) * (10 + idx)

        if idx == 11:
            done = True
        else:
            done = False

        storage.push(sensor_torch, privileged_torch, done)

        print(storage.get_recent_sensor_history())

    sensor_history_minibatch, privileged_minibatch = storage.sample(minibatch_size=3)

    print("\nstorage.privilegeds:\n", storage.privilegeds)
    print("storage.sensors:\n", storage.sensors)
    print("storage.dones[:, None]:\n", storage.dones[:, None])
    print("\nprivileged minibatch:\n", privileged_minibatch)
    print("\nsensor history minibatch:\n", sensor_history_minibatch)
