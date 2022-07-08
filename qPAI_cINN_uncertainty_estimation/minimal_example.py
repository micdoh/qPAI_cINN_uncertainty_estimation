import torch
import torch.nn as nn
from FrEIA.modules import *
from FrEIA.framework import *

if __name__ == "__main__":

    batch_size = 4
    seq_length = 41
    n_features = 1
    lstm_hidden = 100
    inn_hidden = 10
    n_blocks = 6  # No. of invertible blocks in INN

    lstm = nn.LSTM(
        input_size=n_features,  # Input dimensions
        hidden_size=lstm_hidden,  # No. of neurons in gate networks
        batch_first=True,
    )

    class CondNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = lstm
            self.linear = nn.Linear(in_features=lstm_hidden, out_features=1)

        def forward(self, x):
            out = self.lstm(x)[0]
            out = self.linear(out)
            return out  # torch.reshape(out, (batch_size, -1, 1))

    def subnet(dims_in, dims_out):
        return nn.Sequential(
            nn.Linear(dims_in, inn_hidden),
            nn.LeakyReLU(),
            nn.Linear(inn_hidden, inn_hidden),
            nn.LeakyReLU(),
            nn.Linear(inn_hidden, dims_out),
        )

    cond_nn = CondNetwork()
    inn = SequenceINN(
        2,
    )
    for i in range(n_blocks):
        inn.append(AllInOneBlock, cond=0, cond_shape=(2,), subnet_constructor=subnet)

    input = torch.randn(
        batch_size,
        2,
    ).normal_()
    cond = cond_nn(torch.randn(batch_size, 2, 1)).squeeze()
    z = inn(input, c=[cond])
    print(z)
