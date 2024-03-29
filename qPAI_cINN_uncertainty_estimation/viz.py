import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qPAI_cINN_uncertainty_estimation.config as c


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


def save_fig(dir, title):
    if dir:
        dir.mkdir(parents=True, exist_ok=True)
        file = dir / title
        plt.savefig(file.resolve())


def get_relative_iqrs(name, df):
    return np.vstack((df[name] - df["iqr_lower"], df["iqr_upper"] - df[name]))


def plot_training_batch_losses(name, dir=None):
    loss_data = read_files_into_array(name, 'loss_epoch', axis=1)
    fig, ax = plt.subplots()
    ax.set_title('Training Loss', fontsize=12)
    ax.set_xlabel('Batch Number', fontsize=12)
    ax.set_ylabel('NLL Loss', fontsize=12)
    ax.plot(np.arange(loss_data.shape[1]), loss_data.T)
    save_fig(dir, f"{name}_train_batch_loss.png")
    #fig.show()
    plt.clf()


def plot_training_epoch_losses(name, dir=None):
    loss_data = read_files_into_array(name, 'epoch_losses', seq=False, axis=0)
    fig, ax = plt.subplots()
    ax.set_title('Training Loss', fontsize=12)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('NLL Loss', fontsize=12)
    try:
        ax.plot(np.arange(loss_data.shape[1]), loss_data.T)
    except IndexError:
        ax.plot(np.arange(loss_data.shape[0]), loss_data)
    save_fig(dir, f"{name}_train_epoch_loss.png")
    #fig.show()
    plt.clf()


def plot_validation_epoch_losses(name, dir=None):
    loss_data = read_files_into_array(name, 'valid_losses', seq=False, axis=0)
    fig, ax = plt.subplots()
    ax.set_title('Validation Losses', fontsize=12)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('NLL Loss', fontsize=12)
    try:
        ax.plot(np.arange(loss_data.shape[1]), loss_data.T)
    except IndexError:
        ax.plot(np.arange(loss_data.shape[0]), loss_data)
    save_fig(dir, f"{name}_valid_epoch_loss.png")
    #fig.show()
    plt.clf()


def plot_error_bars_preds_vs_g_truth(df, name, measure='median', dir=None):

    x = df["g_truth"] * 100
    y = df[f"{measure}_pred"] * 100

    bins = (np.floor(x/5).astype(int) + 0.5) * 5
    bin_df = pd.concat((y, bins), axis=1)
    bin_df.columns = ['medians', 'bins']
    new_bin_df = bin_df.groupby('bins').median()

    measure_str = f"M{measure[1:]}"
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"{measure_str} predicted sO$_2$ [%]", fontsize=12)
    ax.set_xlabel("Ground truth sO$_2$ [%]", fontsize=12)
    ax.plot(new_bin_df.index, new_bin_df['medians'], color='limegreen', linestyle='-', linewidth=2, zorder=120, label='median of medians')
    ax.plot(np.arange(0, 100), np.arange(0, 100), color='k', linestyle='-', linewidth=2, zorder=100, label='optimum')
    ax.errorbar(
        x, y,
        yerr=get_relative_iqrs(f'{measure}_pred', df) *100,
        elinewidth=0.1,
        capsize=0,
        ecolor='r',
        errorevery=8,
        markevery=8,
        ls='none',
        ms=1,
        fmt='o'
    )
    #ax.set_title(f"{measure_str} sO$_2$ predictions with IQR")
    ax.legend()
    save_fig(dir, f"{name}_{measure}_sO2_preds.png")
    #fig.show()
    plt.clf()


def plot_abs_error(df, name, measure='median', dir=None):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    #ax.set_title(f'Absolute error for {measure} predicted sO$_2$', fontsize=12)
    ax.set_xlabel('Ground truth sO$_2$ [%]', fontsize=12)
    ax.set_ylabel('Absolute error [%]', fontsize=12)
    ax.scatter(df["g_truth"] * 100, df[f"abs_err_{measure}"] * 100, s=1, label=f"{measure} prediction")
    save_fig(dir, f"{name}_abs_error.png")
    #fig.show()
    plt.clf()


def plot_rel_error(df, name, measure='median', dir=None):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xlim(0, 100)
    #ax.set_title(f'Relative error for {measure} predicted sO$_2$', fontsize=12)
    ax.set_xlabel('Ground truth sO$_2$ [%]', fontsize=12)
    ax.set_ylabel('Relative error [%]', fontsize=12)
    ax.scatter(df["g_truth"] * 100, np.abs(df[f"rel_err_{measure}"]) * 100, s=1, label=f"{measure} prediction")
    save_fig(dir, f"{name}_rel_error.png")
    #fig.show()
    plt.clf()

def plot_rel_error_comparison(df1, df2, name1, name2, measure='median', dir=None):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xlim(0, 100)
    #ax.set_title(f'Relative error for {measure} predicted sO$_2$', fontsize=12)
    ax.set_xlabel('Ground truth sO$_2$ [%]', fontsize=12)
    ax.set_ylabel('Relative error [%]', fontsize=12)
    ax.scatter(df1["g_truth"] * 100, np.abs(df1[f"rel_err_{measure}"]) * 100, s=1, alpha=0.5, label=name1)
    ax.scatter(df2["g_truth"] * 100, np.abs(df2[f"rel_err_{measure}"]) * 100, s=1, alpha=0.5, label=name2)
    ax.legend()
    save_fig(dir, f"{name1}_{name2}_rel_error.png")
    #fig.show()
    plt.clf()

def plot_abs_error_comparison(df1, df2, name1, name2, measure='median', dir=None):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xlim(0, 100)
    #ax.set_title(f'Absolute error for {measure} predicted sO$_2$', fontsize=12)
    ax.set_xlabel('Ground truth sO$_2$ [%]', fontsize=12)
    ax.set_ylabel('Absolute error [%]', fontsize=12)
    ax.scatter(df1["g_truth"] * 100, np.abs(df1[f"abs_err_{measure}"]) * 100, s=1, alpha=0.5, label=name1)
    ax.scatter(df2["g_truth"] * 100, np.abs(df2[f"abs_err_{measure}"]) * 100, s=1, alpha=0.5, label=name2)
    ax.legend()
    save_fig(dir, f"{name1}_{name2}_abs_error.png")
    # fig.show()
    plt.clf()


def plot_mean_median_difference(df, name, dir=None):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_title('Difference between mean/median', fontsize=12)
    ax.set_xlabel('Ground truth sO$_2$ [%]', fontsize=12)
    ax.set_ylabel('Mean - Median [%]', fontsize=12)
    ax.scatter(df["g_truth"] * 100, df["mean_pred"] * 100 - df["median_pred"] * 100, s=1)
    save_fig(dir, f"{name}_mean_median_diff.png")
    #fig.show()
    plt.clf()


def plot_calibration_and_uncertainty(calib_df, name, dir=None):
    plt.subplot(2, 1, 1)
    plt.plot(calib_df['confidence'], calib_df['calib_err'])
    plt.ylabel('Calibration error')
    plt.subplot(2, 1, 2)
    plt.plot(calib_df['confidence'], calib_df['uncert_interval'])
    plt.ylabel('Median estimated uncertainty')
    plt.xlabel('Confidence')
    save_fig(dir, f"{name}_calib_uncert.png")
    #plt.show()
    plt.clf()


def plot_calibration_curve(calib_df, name, dir=None):
    calib_fig, calib_ax = plt.subplots()
    calib_ax.set_xlim(0, 100)
    calib_ax.set_ylim(0, 100)
    #calib_ax.set_title('Calibration Curve', fontsize=12)
    calib_ax.set_xlabel('Confidence interval [%]', fontsize=12)
    calib_ax.set_ylabel('Ground truth inliers in interval [%]', fontsize=12)
    calib_ax.plot(calib_df['confidence'] * 100, calib_df['inliers'] * 100, label="calibration curve")
    calib_ax.plot(calib_df['confidence'] * 100, calib_df['confidence'] * 100, color='green', linestyle='--',
                  label="optimum")
    calib_ax.legend()
    save_fig(dir, f"{name}_calib_curve.png")
    #calib_fig.show()
    plt.clf()
