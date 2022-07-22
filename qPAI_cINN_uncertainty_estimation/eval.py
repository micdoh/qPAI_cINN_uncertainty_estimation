import torch
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import qPAI_cINN_uncertainty_estimation.config as c
from qPAI_cINN_uncertainty_estimation.model import WrappedModel, save, load
from qPAI_cINN_uncertainty_estimation.data import prepare_dataloader
from qPAI_cINN_uncertainty_estimation.init_log import init_logger
from qPAI_cINN_uncertainty_estimation.monitoring import config_string


def sample_posterior(data, label):

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


def calibration_error(test_loader, dir=None):
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

        posteriors, means = sample_posterior(x, y)
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

    if dir:
        dir.mkdir(parents=True, exist_ok=True)
        file = dir / "calib_err_plot.png"
        plt.savefig(file.resolve())

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


if __name__ == "__main__":

    start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    output_dir = c.output_dir / start_time

    log_file = c.log_dir / f"{start_time}@test.log"
    logger = init_logger(log_file.resolve())

    test_dataloader = prepare_dataloader(
        c.data_path, c.experiment_name, 'test', c.allowed_datapoints, c.batch_size
    )

    if c.use_default_model:
        model = WrappedModel()
    else:
        model = WrappedModel(
            lstm_dim_in=c.lstm_input_dim,
            lstm_dim_out=c.lstm_hidden,
            fcn_dim_out=c.fcn_dim_out,
            inn_dim_in=c.inn_input_dim,
            cond_length=c.cond_length,
            n_blocks=c.n_blocks,
        )

    if c.use_cuda:
        model.cuda()
    optim = torch.optim.Adam(
        model.params_trainable,
        lr=c.lr,
        betas=c.adam_betas,
        eps=c.eps,
        weight_decay=c.weight_decay,
    )
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=c.gamma)

    saved_state_file = c.output_dir / c.load_date / f"{c.load_date}@cinn.pt"

    load(saved_state_file.resolve(), model, optim)

    test_losses = []

    if c.load_eval_data:
        df_file = c.output_dir / c.load_eval_data_date / f"{c.load_eval_data_date}@dataframe.csv"
        calib_df_file = c.output_dir / c.load_eval_data_date / f"{c.load_eval_data_date}@calib_dataframe.csv"
        df = pd.read_csv(df_file.resolve())
        calib_df = pd.read_csv(calib_df_file.resolve())

    else:

        model.eval()

        df, calib_df = calibration_error(test_dataloader, dir=output_dir)

        if c.save_eval_data:
            output_dir.mkdir(parents=True, exist_ok=True)
            df_file = output_dir / f"{start_time}@dataframe.csv"
            calib_df_file = output_dir / f"{start_time}@calib_dataframe.csv"
            df.to_csv(df_file.resolve())
            calib_df.to_csv(calib_df_file.resolve())
            config_details_file = output_dir / f"{start_time}@test_config.txt"
            logger.info(config_string(config_details_file.resolve()))
            logger.info(f"Data saved to: {df_file.resolve()}")
            logger.info(f"Calibration data saved to: {calib_df_file.resolve()}")

    if c.visualisation:

        def get_relative_iqrs(name):
            return np.vstack((df["iqr_upper"]-df[name], df[name]-df["iqr_lower"]))

        mean_fig, mean_ax = plt.subplots()
        mean_fig.subplots_adjust(top=0.8)
        mean_ax.set_xlim(0, 100)
        mean_ax.set_ylim(0, 100)
        mean_ax.set_ylabel("Mean predicted sO2 (%)")
        mean_ax.set_xlabel("Ground truth sO2 (%)")
        x = df["g_truth"]*100
        y = df["mean_pred"]*100
        mean_ax.plot(np.arange(0, 100), np.arange(0, 100), color='k', linestyle='-', linewidth=2, zorder=100)
        mean_ax.errorbar(
            x, y,
            yerr=get_relative_iqrs('mean_pred')*100,
            elinewidth=0.1,
            capsize=1,
            ecolor='r',
            errorevery=10,
            markevery=10,
            ls='none',
            ms=1,
            fmt='o'
        )
        mean_ax.set_title(f"Mean sO2 predictions of cINN based on 1000 samples")
        mean_fig.show()

        median_fig, median_ax = plt.subplots()
        median_fig.subplots_adjust(top=0.8)
        median_ax.set_xlim(0, 100)
        median_ax.set_ylim(0, 100)
        median_ax.set_ylabel("Median predicted sO2 (%)")
        median_ax.set_xlabel("Ground truth sO2 (%)")
        x = df["g_truth"]*100
        y = df["median_pred"]*100
        median_ax.plot(np.arange(0,100), np.arange(0,100), color='k', linestyle='-', linewidth=2, zorder=100)
        median_ax.errorbar(
            x, y,
            yerr=get_relative_iqrs('median_pred')*100,
            elinewidth=0.1,
            capsize=1,
            ecolor='r',
            errorevery=10,
            markevery=10,
            ls='none',
            ms=1,
            fmt='o',
            zorder=2
        )
        median_ax.set_title(f"Median sO2 predictions of cINN based on 1000 samples")
        median_fig.show()

        abs_err_fig, abs_err_ax = plt.subplots()
        abs_err_ax.set_xlim(0, 100)
        abs_err_ax.set_title('Absolute prediction error for predicted sO2 values')
        abs_err_ax.set_xlabel('Ground truth sO2 (%)')
        abs_err_ax.set_ylabel('Absolute error (%)')
        abs_err_ax.scatter(df["g_truth"]*100, df["abs_err_median"]*100, s=1, label="Median prediction")
        #abs_err_ax.scatter(df["g_truth"]*100, df["abs_err_mean"]*100, s=1, label="Mean prediction")
        #abs_err_ax.legend()
        abs_err_fig.show()

        rel_err_fig, rel_err_ax = plt.subplots()
        rel_err_ax.set_yscale('log')
        rel_err_ax.set_xlim(0, 100)
        rel_err_ax.set_title('Relative prediction error for predicted sO2 values')
        rel_err_ax.set_xlabel('Ground truth sO2 (%)')
        rel_err_ax.set_ylabel('Relative error (%)')
        rel_err_ax.scatter(df["g_truth"]*100, df["rel_err_median"]*100, s=1, label="Median prediction")
        #rel_err_ax.scatter(df["g_truth"]*100, df["rel_err_mean"]*100, s=1, label="Mean prediction")
        #rel_err_ax.legend()
        rel_err_fig.show()
        
        pred_diff_fig, pred_diff_ax = plt.subplots()
        pred_diff_ax.set_xlim(0, 100)
        pred_diff_ax.set_title('Difference between Mean/Median')
        pred_diff_ax.set_xlabel('Ground truth sO2 (%)')
        pred_diff_ax.set_ylabel('Median - Mean (%)')
        pred_diff_ax.scatter(df["g_truth"]*100, df["mean_pred"]*100-df["median_pred"]*100, s=1, label="Median prediction")
        pred_diff_fig.show()

        plt.subplot(2, 1, 1)
        plt.plot(calib_df['confidence'], calib_df['calib_err'])
        plt.ylabel('Calibration error')
        plt.subplot(2, 1, 2)
        plt.plot(calib_df['confidence'], calib_df['uncert_interval'])
        plt.ylabel('Median estimated uncertainty')
        plt.xlabel('Confidence')
        plt.show()

        print(F'Median calibration error:               {np.median(np.abs(calib_df["calib_err"]))*100:.1f}%')
        print(F'Calibration error at 68% confidence:    {calib_df["calib_err"][68]*100:.1f}%')
        print(F'Med. est. uncertainty at 68% conf.:     {calib_df["uncert_interval"][68]*100:.1f}%')

        #ranks = df.rank(pct=True)
        #close_to_median = abs(ranks - 0.5)
        #idx = close_to_median.idxmin()['rel_err_median']
        #med = df['rel_err_median'][idx]
        #median_err = np.abs(df['rel_err_median'].median()*100)
        iqr_median_err = np.abs(df['rel_err_median']).quantile([0.25, 0.5, 0.75])*100
        #median_iqr_upper = df['iqr_upper'][idx]
        #median_iqr_lower = df['iqr_lower'][idx]
        print(F'Median relative error and IQR: {iqr_median_err[0.5]:.1f}% \t ({iqr_median_err[0.25]:.1f}%, {iqr_median_err[0.75]:.1f}%)')


