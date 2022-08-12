import matplotlib.pyplot as plt
import numpy as np
import qPAI_cINN_uncertainty_estimation.config as c

if __name__ == "__main__":
    #date = '2022-07-12_11_08_52'  # Initial model
    #date = '2022-07-14_17_52_20'  # Larger model
    #date = '2022-07-20_17_16_01'  # Wider model, less blocks, no FCN, 10 wavelengths
    #date = '2022-07-21_09_39_35'  # Wider model, less blocks, with FCN, 10 wavelengths
    date = '2022-07-24_21_29_34'  # Wider model, less blocks, with FCN, 40 wavelengths
    date = '2022-07-25_16_13_12'  # Wider model, less blocks, without FCN, flexi-train

    loss_data = read_files_into_array(date, 'loss_epoch', axis=0)
    fig1, ax1 = plt.subplots()
    ax1.set_title('Training Loss - Batches')
    ax1.set_xlabel('Batches')
    ax1.set_ylabel('NLL Loss')
    for i in range(len(loss_data)):
        x = np.arange(loss_data.shape[1])
        y = loss_data[i]
        ax1.plot(x, y)
