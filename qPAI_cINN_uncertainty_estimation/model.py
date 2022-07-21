import logging
import torch
import torch.nn as nn
from FrEIA.modules import *
from FrEIA.framework import *
import qPAI_cINN_uncertainty_estimation.config as c


logger = logging.getLogger(__name__)


def subnet(dims_in, dims_out):
    return nn.Sequential(
        nn.Linear(dims_in, c.inn_hidden),
        nn.LeakyReLU(),
        nn.Linear(c.inn_hidden, c.inn_hidden),
        nn.LeakyReLU(),
        nn.Linear(c.inn_hidden, c.inn_hidden),
        nn.LeakyReLU(),
        nn.Linear(c.inn_hidden, dims_out),
    )


class CondNetwork(nn.Module):
    def __init__(self, lstm_dim_in: int, lstm_dim_out: int, fcn_dim_out: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=lstm_dim_in,  # Input dimensions
            hidden_size=lstm_dim_out,  # No. of neurons in gate networks
            batch_first=True,
        )
        if c.use_fcn_layer:
            self.linear = nn.Linear(in_features=lstm_dim_out, out_features=fcn_dim_out)

    def forward(self, x):
        out = self.lstm(x)[0]
        out = self.linear(out) if c.use_fcn_layer else \
            torch.reshape(out, (-1, c.cond_length))
        return out


class WrappedModel(nn.Module):
    def __init__(
        self,
        lstm_dim_in: int = 2,
        lstm_dim_out: int = 100,
        fcn_dim_out: int = 1,
        inn_dim_in: int = 2,
        cond_length: int = 41,
        n_blocks: int = 6,
    ):
        super().__init__()
        self.cond_network = CondNetwork(lstm_dim_in, lstm_dim_out, fcn_dim_out)
        self.inn = self.build_inn(inn_dim_in, cond_length, n_blocks)
        self.params_trainable = list(
            filter(lambda p: p.requires_grad, self.inn.parameters())
        ) + list(self.cond_network.parameters())

    def build_inn(self, inn_dim_in: int, cond_length: int, n_blocks: int):
        inn = SequenceINN(
            inn_dim_in,
        )
        for i in range(n_blocks):
            inn.append(
                AllInOneBlock,
                cond=0,
                cond_shape=(cond_length,),
                subnet_constructor=subnet,
            )
        return inn

    def get_condition(self, data):
        return self.cond_network(data)

    def forward(self, data, label):
        cond = self.get_condition(data)
        z, log_jac_det = self.inn(label, [cond])
        return z, log_jac_det

    def reverse_sample(self, data, z):
        cond = self.get_condition(data)
        return self.inn(z, [cond], rev=True, jac=False)


def save(name, optim, model):
    torch.save({"opt": optim.state_dict(), "net": model.state_dict()}, name)


def load(filepath, model, optim):
    state_dicts = torch.load(filepath, map_location=c.device)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    model.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except Exception as e:
        logger.info(f'Cannot load optimizer: {e}')
