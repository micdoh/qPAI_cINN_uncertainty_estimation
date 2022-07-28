import torch
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import qPAI_cINN_uncertainty_estimation.config as c
from qPAI_cINN_uncertainty_estimation.model import WrappedModel, save, load, init_model
from qPAI_cINN_uncertainty_estimation.data import prepare_dataloader
from qPAI_cINN_uncertainty_estimation.init_log import init_logger
from qPAI_cINN_uncertainty_estimation.monitoring import config_string


def save_fig(dir, title):
    if dir:
        dir.mkdir(parents=True, exist_ok=True)
        file = dir / title
        plt.savefig(file.resolve())


def get_relative_iqrs(name, df):
    return np.vstack((df["iqr_upper"] - df[name], df[name] - df["iqr_lower"]))


def plot_error_bars_preds_vs_g_truth(df, measure='median', dir=None):
    measure_str = f"M{measure[1:]}"
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"{measure_str} predicted sO2 (%)")
    ax.set_xlabel("Ground truth sO2 (%)")
    x = df["g_truth" ] *100
    y = df[f"{measure}_pred" ] *100
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
    save_fig(dir, f"{measure}_sO2_predictions.png")
    fig.show()


def plot_abs_error(df, measure='median', dir=None):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_title(f'Absolute error for {measure} predicted sO2 values')
    ax.set_xlabel('Ground truth sO2 (%)')
    ax.set_ylabel('Absolute error (%)')
    ax.scatter(df["g_truth"] * 100, df[f"abs_err_{measure}"] * 100, s=1, label=f"{measure} prediction")
    save_fig(dir, "abs_error.png")
    fig.show()


def plot_rel_error(df, measure='median', dir=None):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xlim(0, 100)
    ax.set_title(f'Relative error for {measure} predicted sO2 values')
    ax.set_xlabel('Ground truth sO2 (%)')
    ax.set_ylabel('Relative error (%)')
    ax.scatter(df["g_truth"] * 100, df[f"rel_err_{measure}"] * 100, s=1, label=f"{measure} prediction")
    save_fig(dir, "rel_error.png")
    fig.show()


def plot_mean_median_difference(df,  dir=None):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_title('Difference between Mean/Median')
    ax.set_xlabel('Ground truth sO2 (%)')
    ax.set_ylabel('Mean - Median (%)')
    ax.scatter(df["g_truth"] * 100, df["mean_pred"] * 100 - df["median_pred"] * 100, s=1)
    save_fig(dir, "mean_median_diff.png")
    fig.show()


def plot_calibration_and_uncertainty(calib_df,  dir=None):
    plt.subplot(2, 1, 1)
    plt.plot(calib_df['confidence'], calib_df['calib_err'])
    plt.ylabel('Calibration error')
    plt.subplot(2, 1, 2)
    plt.plot(calib_df['confidence'], calib_df['uncert_interval'])
    plt.ylabel('Median estimated uncertainty')
    plt.xlabel('Confidence')
    save_fig(dir, "calibration_uncertainty.png")
    plt.show()


def plot_calibration_curve(calib_df, dir=None):
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
    save_fig(dir, "calibration_curve.png")
    calib_fig.show()