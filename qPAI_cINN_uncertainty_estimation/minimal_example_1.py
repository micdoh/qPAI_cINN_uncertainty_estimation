import torch
import torch.nn as nn
from FrEIA.modules import *
from FrEIA.framework import *

if __name__ == "__main__":

    batch_size = 3
    seq_length = 41
    n_features = 1
    inn_input_dim = 2
    lstm_hidden = 100
    inn_hidden = 128
    n_blocks = 6  # No. of invertible blocks in INN

    lstm = nn.LSTM(
        input_size=n_features, # Input dimensions
        hidden_size=lstm_hidden, # No. of neurons in gate networks
        batch_first=True
    )

    class CondNetwork(nn.Module):
        def __init__(self, lstm_dim_in: int, lstm_dim_out: int, fcn_dim_out: int):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=lstm_dim_in, # Input dimensions
                hidden_size=lstm_dim_out, # No. of neurons in gate networks
                batch_first=True
            )
            self.linear = nn.Linear(
                    in_features=lstm_dim_out,
                    out_features=fcn_dim_out
            )

        def forward(self, x):
            out = self.lstm(x)[0]
            out = self.linear(out)
            return out  # torch.reshape(out, (batch_size, -1, 1))

    def subnet(dims_in, dims_out):
        return nn.Sequential(nn.Linear(dims_in, inn_hidden), nn.LeakyReLU(),
                             nn.Linear(inn_hidden,  inn_hidden), nn.LeakyReLU(),
                             nn.Linear(inn_hidden, dims_out))


    class WrappedModel(nn.Module):

        def __init__(self, lstm_dim_in: int = 1, lstm_dim_out: int = 100, fcn_dim_out: int = 1, inn_dim_in: int = 2, cond_length: int = 41):
            super().__init__()
            self.cond_network = CondNetwork(lstm_dim_in, lstm_dim_out, fcn_dim_out)
            self.inn = self.build_inn(inn_dim_in, cond_length)

        def build_inn(self, inn_dim_in: int, cond_length: int):
            inn = SequenceINN(inn_dim_in,)
            for i in range(n_blocks):
                inn.append(
                    AllInOneBlock,
                    cond=0,
                    cond_shape=(cond_length,),
                    subnet_constructor=subnet
                )
            return inn

        def get_condition(self, data):
            cond = self.cond_network(data).squeeze()
            # cond_view = cond.view(cond.size(0), -1)  # TODO - Not sure which of view or squeeze is better to use... maybe either is ok
            return cond

        def forward(self, data, label):
            cond = self.get_condition(data)
            # TODO - Does label need to be extended to length 41, to match condition?
            # TODO - Might be better to reshape cond to be able to join input
            z, log_jac_det = self.inn(label, [cond])
            #zz = sum(torch.sum(o ** 2, dim=1) for o in z)
            #jac = self.inn.jacobian(run_forward=False)

            return z, log_jac_det

        def reverse_sample(self, z, cond):
            return self.inn(z, cond, rev=True)

    model = WrappedModel()

    label = torch.randn(batch_size, inn_input_dim, ).normal_()  # (batch, 2, 1) must be the shape, to allow splitting
    data = torch.randn(batch_size, seq_length, 1)
    z, log_jac_det = model(data, label)
    nll = torch.mean(z ** 2) / 2 - torch.mean(log_jac_det) / 40  # total dimensions of data
    print(z, log_jac_det)

# TODO - build training loop and load in data from dataloader, then actually train!