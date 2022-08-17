import matplotlib.pyplot as plt
import numpy as np
import qPAI_cINN_uncertainty_estimation.config as c
from qPAI_cINN_uncertainty_estimation.viz import (
    plot_training_epoch_losses,
    plot_validation_epoch_losses,
    plot_training_batch_losses,
)


def read_files_into_array(date: str,
                          name: str,
                          suffix: str = '.npy',
                          seq: bool = True,
                          axis: int = 1,
                          start_index: int = 1
                          ):
    """Read files into a numpy array
    If seq = True, read a series of files and add an index"""
    dir = c.output_dir / date
    print(dir)
    index = start_index
    if seq:
        while True:
            try:
                file = dir / f"{date}@{name}_{index}{suffix}"
                with open(file.resolve(), 'rb') as f:
                    data = np.load(f)
                    if index > start_index:
                        data = np.concatenate((old_data, data), axis=axis)
                    old_data = data
                    index += 1
            except FileNotFoundError:
                print(f'File not found: {file.resolve()}')
                break
    else:
        file = dir / f"{date}@{name}{suffix}"
        with open(file.resolve(), 'rb') as f:
            data = np.load(f)
    return data



if __name__ == "__main__":
    #date = '2022-07-12_11_08_52'  # Initial model
    #date = '2022-07-14_17_52_20'  # Larger model
    #date = '2022-07-20_17_16_01'  # Wider model, less blocks, no FCN, 10 wavelengths
    #date = '2022-07-21_09_39_35'  # Wider model, less blocks, with FCN, 10 wavelengths
    date = '2022-07-24_21_29_34'  # Wider model, less blocks, with FCN, 40 wavelengths
    date = '2022-07-25_16_13_12'  # Wider model, less blocks, without FCN, flexi-train

    #date = '2022-07-12_11_08_52'  # Initial model
    #date = '2022-07-14_17_52_20'  # Larger model
    #date = '2022-07-20_17_16_01'  # Wider model, less blocks, no FCN, 10 wavelengths
    #date = '2022-07-21_09_39_35'  # Wider model, less blocks, with FCN, 10 wavelengths
    date = '2022-07-24_21_29_34'  # Wider model, less blocks, with FCN, 40 wavelengths

    output_dir = c.output_dir / date

    plot_training_batch_losses(date, dir=output_dir)
    plot_training_epoch_losses(date, dir=output_dir)
    plot_validation_epoch_losses(date, dir=output_dir)

    #loss_data = read_files_into_array(date, 'loss_epoch', axis=1)
    #fig, ax = plt.subplots()
    #ax.set_title('Training Loss - Batches')
    #ax.set_xlabel('Batches')
    #ax.set_ylabel('NLL Loss')
    #ax.plot(np.arange(loss_data.shape[1]), loss_data.T)

    #loss_data = read_files_into_array(date, 'loss_epoch', axis=0)
    #fig1, ax1 = plt.subplots()
    #ax1.set_title('Training Loss - Batches')
    #ax1.set_xlabel('Batches')
    #ax1.set_ylabel('NLL Loss')
    #for i in range(len(loss_data)):
    #    x = np.arange(loss_data.shape[1])
    #    y = loss_data[i]
    #    ax1.plot(x, y)

    #loss_data = read_files_into_array(date, 'epoch_losses', seq=False, axis=0)
    #fig2, ax2 = plt.subplots()
    #ax2.set_title('Training Loss - Epochs')
    #ax2.set_xlabel('Epochs')
    #ax2.set_ylabel('NLL Loss')
    #ax2.plot(np.arange(loss_data.shape[1]), loss_data.T)

    #loss_data = read_files_into_array(date, 'valid_losses', seq=False, axis=0)
    #fig3, ax3 = plt.subplots()
    #ax3.set_title('Validation Losses')
    #ax3.set_xlabel('Epochs')
    #ax3.set_ylabel('NLL Loss')
    #ax3.plot(np.arange(loss_data.shape[1]), loss_data.T)

    plt.show()
