import numpy as np
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


def save_fig(dir, title):
    if dir:
        dir.mkdir(parents=True, exist_ok=True)
        file = dir / title
        plt.savefig(file.resolve())


def get_relative_iqrs(name, df):
    return np.vstack((df["iqr_upper"] - df[name], df[name] - df["iqr_lower"]))


def plot_training_batch_losses(name, dir=None):
    loss_data = read_files_into_array(name, 'loss_epoch', axis=1)
    fig, ax = plt.subplots()
    ax.set_title('Training Loss')
    ax.set_xlabel('Batch Number')
    ax.set_ylabel('NLL Loss')
    ax.plot(np.arange(loss_data.shape[1]), loss_data.T)
    save_fig(dir, f"{name}_train_batch_loss.png")
    fig.show()


def plot_training_epoch_losses(name, dir=None):
    loss_data = read_files_into_array(name, 'epoch_losses', seq=False, axis=0)
    fig, ax = plt.subplots()
    ax.set_title('Training Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('NLL Loss')
    print(loss_data.shape)
    try:
        ax.plot(np.arange(loss_data.shape[1]), loss_data.T)
    except IndexError:
        ax.plot(np.arange(loss_data.shape[0]), loss_data)
    save_fig(dir, f"{name}_train_epoch_loss.png")
    fig.show()


def plot_validation_epoch_losses(name, dir=None):
    loss_data = read_files_into_array(name, 'valid_losses', seq=False, axis=0)
    fig, ax = plt.subplots()
    ax.set_title('Validation Losses')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('NLL Loss')
    ax.plot(np.arange(loss_data.shape[1]), loss_data.T)
    save_fig(dir, f"{name}_valid_epoch_loss.png")
    fig.show()


def plot_error_bars_preds_vs_g_truth(df, name, measure='median', dir=None):
    measure_str = f"M{measure[1:]}"
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"{measure_str} predicted sO2 (%)")
    ax.set_xlabel("Ground truth sO2 (%)")
    x = df["g_truth" ] * 100
    y = df[f"{measure}_pred" ] * 100
    ax.plot(np.arange(0, 100), np.arange(0, 100), color='k', linestyle='-', linewidth=2, zorder=100)
    ax.errorbar(
        x, y,
        yerr=get_relative_iqrs(f'{measure}_pred', df) *100,
        elinewidth=0.1,
        capsize=1,
        ecolor='r',
        errorevery=10,
        markevery=10,
        ls='none',
        ms=1,
        fmt='o'
    )
    ax.set_title(f"{measure_str} sO2 predictions of cINN based on 1000 samples")
    save_fig(dir, f"{name}_{measure}_sO2_preds.png")
    fig.show()


def plot_abs_error(df, name, measure='median', dir=None):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_title(f'Absolute error for {measure} predicted sO2 values')
    ax.set_xlabel('Ground truth sO2 (%)')
    ax.set_ylabel('Absolute error (%)')
    ax.scatter(df["g_truth"] * 100, df[f"abs_err_{measure}"] * 100, s=1, label=f"{measure} prediction")
    save_fig(dir, f"{name}_abs_error.png")
    fig.show()


def plot_rel_error(df, name, measure='median', dir=None):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xlim(0, 100)
    ax.set_title(f'Relative error for {measure} predicted sO2 values')
    ax.set_xlabel('Ground truth sO2 (%)')
    ax.set_ylabel('Relative error (%)')
    ax.scatter(df["g_truth"] * 100, df[f"rel_err_{measure}"] * 100, s=1, label=f"{measure} prediction")
    save_fig(dir, f"{name}_rel_error.png")
    fig.show()


def plot_mean_median_difference(df, name, dir=None):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_title('Difference between Mean/Median')
    ax.set_xlabel('Ground truth sO2 (%)')
    ax.set_ylabel('Mean - Median (%)')
    ax.scatter(df["g_truth"] * 100, df["mean_pred"] * 100 - df["median_pred"] * 100, s=1)
    save_fig(dir, f"{name}_mean_median_diff.png")
    fig.show()


def plot_calibration_and_uncertainty(calib_df, name, dir=None):
    plt.subplot(2, 1, 1)
    plt.plot(calib_df['confidence'], calib_df['calib_err'])
    plt.ylabel('Calibration error')
    plt.subplot(2, 1, 2)
    plt.plot(calib_df['confidence'], calib_df['uncert_interval'])
    plt.ylabel('Median estimated uncertainty')
    plt.xlabel('Confidence')
    save_fig(dir, f"{name}_calib_uncert.png")
    plt.show()


def plot_calibration_curve(calib_df, name, dir=None):
    calib_fig, calib_ax = plt.subplots()
    calib_ax.set_xlim(0, 100)
    calib_ax.set_ylim(0, 100)
    calib_ax.set_title('cINN Calibration Curve')
    calib_ax.set_xlabel('Confidence interval (%)')
    calib_ax.set_ylabel('Ground truth inliers in interval (%)')
    calib_ax.plot(calib_df['confidence'] * 100, calib_df['inliers'] * 100, label="Calibration curve")
    calib_ax.plot(calib_df['confidence'] * 100, calib_df['confidence'] * 100, color='green', linestyle='--',
                  label="Optimal")
    calib_ax.legend()
    save_fig(dir, f"{name}_calib_curve.png")
    calib_fig.show()
