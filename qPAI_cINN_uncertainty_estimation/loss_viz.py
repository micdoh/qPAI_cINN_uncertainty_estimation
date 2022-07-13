from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import qPAI_cINN_uncertainty_estimation.config as c


def read_files_into_array(date: str, name: str, suffix: str = '.npy', seq: bool = True, axis: int = 1):
    """Read files into a numpy array
    If seq = True, read a series of files and add an index"""
    dir = c.output_dir / date
    index = 0
    if seq:
        while True:
            try:
                file = dir / f"{date}@{name}_{index}{suffix}"
                with open(file.resolve(), 'rb') as f:
                    data = np.load(f)
                    if index > 0:
                        data = np.concatenate((old_data, data), axis=axis)
                    old_data = data
                    index += 1
            except FileNotFoundError:
                print('filenotound')
                break
    else:
        file = dir / f"{date}@{name}{suffix}"
        with open(file.resolve(), 'rb') as f:
            data = np.load(f)
    return data


#def viz_epoch_loss(fig, date: str):


#def viz_


if __name__ == "__main__":
    loss_data = read_files_into_array('2022-07-12_11_08_52', 'loss_epoch', axis=1)
    fig, ax = plt.subplots()
    ax.set_title('Batch Losses')
    ax.set_xlabel('Batches')
    ax.set_ylabel('NLL Loss')
    ax.plot(np.arange(loss_data.shape[1]), loss_data.T)

    loss_data = read_files_into_array('2022-07-12_11_08_52', 'loss_epoch', axis=0)
    fig1, ax1 = plt.subplots()
    ax1.set_title('Batch Losses')
    ax1.set_xlabel('Batches')
    ax1.set_ylabel('NLL Loss')
    for i in range(len(loss_data)):
        x = np.arange(loss_data.shape[1])
        y = loss_data[i]
        ax1.plot(x, y)

    loss_data = read_files_into_array('2022-07-12_11_08_52', 'epoch_losses', seq=False, axis=0)
    fig2, ax2 = plt.subplots()
    ax2.set_title('Epoch Losses')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('NLL Loss')
    ax2.plot(np.arange(loss_data.shape[1]), loss_data.T)
    plt.show()
