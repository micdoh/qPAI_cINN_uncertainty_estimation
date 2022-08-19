import torch
import pickle
import sys
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
from qPAI_cINN_uncertainty_estimation.viz import (
    plot_calibration_and_uncertainty,
    plot_calibration_curve,
    plot_mean_median_difference,
    plot_abs_error,
    plot_rel_error,
    plot_error_bars_preds_vs_g_truth,
    plot_training_epoch_losses,
    plot_validation_epoch_losses,
    plot_training_batch_losses,
)


def sample_posterior(model, data, label):

    for i in range(c.n_samples):
        rev_inputs = torch.randn_like(label)
        with torch.no_grad():
            x_sample, _ = model.reverse_sample(data, rev_inputs)
            x_sample = x_sample.mean(dim=1).unsqueeze(dim=1)

        x_samples = torch.cat((x_samples, x_sample), dim=1) if i != 0 else x_sample

    rev_inputs = torch.zeros_like(label)
    with torch.no_grad():
        means, _ = model.reverse_sample(data, rev_inputs)

    return x_samples, means


def evaluation_and_calibration(model, test_loader, dir=None, eval_str=''):
    # how many different confidences to look at
    n_steps = 100

    q_values = []
    confidences = np.linspace(0., 1., n_steps + 1, endpoint=False)[1:]
    uncert_intervals = [[] for i in range(n_steps)]
    inliers = [[] for i in range(n_steps)]

    labels = np.array([])
    mean_pred = np.array([])
    median_pred = np.array([])
    iqr_uppers = np.array([])
    iqr_lowers = np.array([])

    for conf in confidences:
        q_low = 0.5 * (1 - conf)
        q_high = 0.5 * (1 + conf)
        q_values += [q_low, q_high]

    for x, y in tqdm(iter(test_loader),
                total=len(test_loader),
                leave=True,
                position=0,
                mininterval=1.0,
                ncols=83,):

        x, y = x.to(c.device), y.to(c.device)

        posteriors, means = sample_posterior(model, x, y)
        means = means.mean(dim=1).unsqueeze(dim=1)
        g_truths = y.mean(dim=1).unsqueeze(dim=1)

        for post, g_truth, mean in zip(posteriors, g_truths, means):

            post = post.data.cpu().numpy()
            g_truth = g_truth.data.cpu().numpy()
            mean = mean.data.cpu().numpy()
            median = np.median(post)
            iqr_upper, iqr_lower = np.percentile(post, [75, 25])
            x_margins = list(np.quantile(post, q_values))

            for i in range(n_steps):
                x_low, x_high = x_margins.pop(0), x_margins.pop(0)

                uncert_intervals[i].append(x_high - x_low)
                inliers[i].append(int(x_low < g_truth < x_high))
                
            labels = np.append(labels, g_truth)
            mean_pred = np.append(mean_pred, mean)
            median_pred = np.append(median_pred, median)
            iqr_uppers = np.append(iqr_uppers, iqr_upper)
            iqr_lowers = np.append(iqr_lowers, iqr_lower)
            
    abs_errs_mean = mean_pred - labels
    rel_errs_mean = abs_errs_mean / labels
    abs_errs_median = median_pred - labels
    rel_errs_median = abs_errs_median / labels
    iqrs = iqr_uppers - iqr_lowers

    inliers = np.mean(inliers, axis=1)
    uncert_intervals = np.median(uncert_intervals, axis=1)
    calib_err = inliers - confidences

    calib_df = pd.DataFrame({"confidence": confidences,
                             "uncert_interval": uncert_intervals,
                             "inliers": inliers,
                             "calib_err": calib_err,
                             })

    # Load everything into a dataframe
    df = pd.DataFrame({"mean_pred": mean_pred,
                       "median_pred": median_pred,
                       "g_truth": labels,
                       "iqr_upper": iqr_uppers,
                       "iqr_lower": iqr_lowers,
                       "iqr": iqrs,
                       "abs_err_mean": abs_errs_mean,
                       "rel_err_mean": rel_errs_mean,
                       "abs_err_median": abs_errs_median,
                       "rel_err_median": rel_errs_median,
                       })
    df.sort_values(by=["g_truth"], ascending=False, inplace=True)

    return df, calib_df


def eval_model(
        model_name=None,
        allowed_datapoints=c.allowed_datapoints,
        experiment_name=c.experiment_name,
        eval_str='',
):

    start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    name = model_name if model_name else start_time

    output_dir = c.output_dir / name

    log_file = c.log_dir / f"{name}{eval_str}@test.log"
    logger = init_logger(log_file.resolve(), log_file.stem)

    if c.load_eval_data:

        #df_file = c.output_dir / c.load_eval_data_date / f"{c.load_eval_data_date}@dataframe.csv"
        #calib_df_file = c.output_dir / c.load_eval_data_date / f"{c.load_eval_data_date}@calib_dataframe.csv"
        output_dir = output_dir / eval_str
        df_file = output_dir / f"{name}@dataframe.csv"
        calib_df_file = output_dir / f"{name}@calib_dataframe.csv"
        df = pd.read_csv(df_file.resolve())
        calib_df = pd.read_csv(calib_df_file.resolve())

    else:

        test_dataloader, _ = prepare_dataloader(
            c.data_path, experiment_name, 'test', allowed_datapoints, c.batch_size, sampler=c.balance_dataset
        )

        model, optim, weight_scheduler = init_model()

        saved_state_file = output_dir / f"{name}@cinn.pt"

        load(saved_state_file.resolve(), model, optim)

        model.eval()

        output_dir = output_dir / eval_str

        df, calib_df = evaluation_and_calibration(model, test_dataloader, dir=output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        df_file = output_dir / f"{name}@dataframe.csv"
        calib_df_file = output_dir / f"{name}@calib_dataframe.csv"
        df.to_csv(df_file.resolve())
        calib_df.to_csv(calib_df_file.resolve())
        config_details_file = output_dir / f"{name}@test_config.txt"
        logger.info(config_string(config_details_file.resolve()))
        logger.info(f"Data saved to: {df_file.resolve()}")
        logger.info(f"Calibration data saved to: {calib_df_file.resolve()}")

    plot_calibration_and_uncertainty(calib_df, f'{name}_{allowed_datapoints}', dir=output_dir)
    plot_calibration_curve(calib_df, f'{name}_{allowed_datapoints}', dir=output_dir)
    plot_mean_median_difference(df, f'{name}_{allowed_datapoints}', dir=output_dir)
    plot_abs_error(df, f'{name}_{allowed_datapoints}', dir=output_dir)
    plot_rel_error(df, f'{name}_{allowed_datapoints}', dir=output_dir)
    plot_error_bars_preds_vs_g_truth(df, f'{name}_{allowed_datapoints}', dir=output_dir)

    iqr_median_err = np.abs(df['rel_err_median']).quantile([0.25, 0.5, 0.75]) * 100
    median_calib_err = np.median(np.abs(calib_df["calib_err"]))
    calib_err_68 = calib_df["calib_err"][68]
    med_uncert_68 = calib_df["uncert_interval"][68]

    logger.info(F'Model evaluated: {model_name}')
    logger.info(F'Median calibration error:               {median_calib_err*100:.1f}%')
    logger.info(F'Calibration error at 68% confidence:    {calib_err_68*100:.1f}%')
    logger.info(F'Median est. uncertainty at 68% conf.:     {med_uncert_68*100:.1f}%')
    logger.info(F'Median relative error and IQR: {iqr_median_err[0.5]:.1f}% \t ({iqr_median_err[0.25]:.1f}%, {iqr_median_err[0.75]:.1f}%)')

    row = {
        'name': name,
        'partitioned_eval': eval_str.split('_')[-2],
        'sparsity_eval': eval_str.split('_')[-1],
        'med_calib_err': median_calib_err,
        'calib_err_68': calib_err_68,
        'med_uncert_68': med_uncert_68,
        'med_rel_err': iqr_median_err[0.5],
        'iqr_lower': iqr_median_err[0.25],
        'iqr_upper': iqr_median_err[0.75],
    } if eval_str else {}

    return df, calib_df, row


if __name__ == "__main__":
    model_name = sys.argv[1]
    eval_model(model_name)
    output_dir = c.output_dir / model_name
    plot_training_batch_losses(model_name, dir=output_dir)
    plot_training_epoch_losses(model_name, dir=output_dir)
    plot_validation_epoch_losses(model_name, dir=output_dir)
