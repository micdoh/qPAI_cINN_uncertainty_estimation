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

    rev_inputs = torch.randn_like(label)

    for i in range(c.n_samples):
        with torch.no_grad():
            x_sample, _ = model.reverse_sample(data, rev_inputs)
            x_sample = x_sample.mean(dim=1).unsqueeze(dim=1)

        x_samples = torch.cat((x_samples, x_sample), dim=1) if i != 0 else x_sample

    return x_samples


def calibration_error(test_loader, dir=None):
    # how many different confidences to look at
    n_steps = 100

    q_values = []
    confidences = np.linspace(0., 1., n_steps + 1, endpoint=False)[1:]
    uncert_intervals = [[] for i in range(n_steps)]
    inliers = [[] for i in range(n_steps)]

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

        posteriors = sample_posterior(x, y)
        y = y.mean(dim=1).unsqueeze(dim=1)

        for post, g_truth in zip(posteriors, y):

            post = post.data.cpu().numpy()
            g_truth = g_truth.data.cpu().numpy()
            x_margins = list(np.quantile(post, q_values))

            for i in range(n_steps):
                x_low, x_high = x_margins.pop(0), x_margins.pop(0)

                uncert_intervals[i].append(x_high - x_low)
                inliers[i].append(int(x_low < g_truth < x_high))

    inliers = np.mean(inliers, axis=1)
    uncert_intervals = np.median(uncert_intervals, axis=1)
    calib_err = inliers - confidences

    print(F'Median calibration error:               {np.median(np.abs(calib_err))}')
    print(F'Calibration error at 68% confidence:    {calib_err[68]}')
    print(F'Med. est. uncertainty at 68% conf.:     {uncert_intervals[68]}')

    plt.subplot(2, 1, 1)
    plt.plot(confidences, calib_err)
    plt.ylabel('Calibration error')

    plt.subplot(2, 1, 2)
    plt.plot(confidences, uncert_intervals)
    plt.ylabel('Median estimated uncertainty')
    plt.xlabel('Confidence')

    if dir:
        dir.mkdir(parents=True, exist_ok=True)
        file = dir / "calib_err_plot.png"
        plt.savefig(file.resolve())

    plt.show()


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

    calibration_error(test_dataloader, dir=output_dir)
    exit

    if c.load_eval_data:
        df_file = c.output_dir / c.load_eval_data_date / f"{c.load_eval_data_date}@dataframe.csv"
        df = pd.read_csv(df_file.resolve())

    else:

        labels = np.array([])
        preds = np.array([])
        errors = np.array([])
        stdevs = np.array([])

        model.eval()  # Optional when not using Model Specific layer
        for i_val, (data, label) in enumerate(test_dataloader):
            # Transfer Data to GPU if available
            data, label = data.to(c.device), label.to(c.device)

            if c.sample_posterior:

                for i_mean in range(c.n_means):

                    for i_sample in range(c.n_samples):
                        # Define input for reverse pass
                        rev_inputs = torch.randn_like(label)
                        # Forward Pass
                        with torch.no_grad():  # Is this necessary after model.eval()
                            z, _ = model.reverse_sample(data, rev_inputs)

                        z = z.mean(dim=1).unsqueeze(dim=1)  # Take mean of the two features/estimates
                        z_samples = torch.cat((z_samples, z), dim=1) if i_sample != 0 else z

                    z_mean = z_samples.mean(dim=1).unsqueeze(dim=1)
                    z_means = torch.cat((z_means, z_mean), dim=1) if i_mean != 0 else z_mean

                z_pred = z_means.mean(dim=1).detach().cpu().numpy()
                z_stdev = z_means.std(dim=1, unbiased=True).detach().cpu().numpy()
                #for samples in z_samples:
                #    fig, ax = plt.subplots()
                #    ax.set_title('Posterior Distribution of sO2 Values obtained from \n'
                #                 'random sampling of Gaussian latent space')
                #    ax.set_xlabel('sO2')
                #    ax.set_ylabel('N Samples')
                #    ax.hist(samples, np.arange(0, 1, 0.05))
                #    fig.show()

            else:
                # Sample with tensors (0,0) for mean, (-1,-1) for lower stdev, (1, 1) for upper stdev
                mean_rev_inputs = torch.zeros_like(label)
                stdev_rev_inputs = torch.ones_like(label)

                with torch.no_grad():
                    z_pred, _ = model.reverse_sample(data, mean_rev_inputs)
                    z_stdev, _ = model.reverse_sample(data, stdev_rev_inputs)

                z_pred = z_pred.mean(dim=1).detach().numpy()
                z_stdev = z_stdev.mean(dim=1).detach().numpy()
                stdev = np.subtract(z_stdev, z_pred)

            label = label.mean(dim=1).detach().cpu().numpy()

            err = np.subtract(z_pred, label)
            errors = np.append(errors, err)
            labels = np.append(labels, label)
            preds = np.append(preds, z_pred)
            stdevs = np.append(stdevs, stdev)

        # Load everything into a dataframe
        df = pd.DataFrame({"errors": errors,
                           "labels": labels,
                           "preds": preds,
                           "stdevs": stdevs,
                           })
        df.sort_values(by=["labels"], ascending=False, inplace=True)

        if c.save_eval_data:
            output_dir.mkdir(parents=True, exist_ok=True)
            df_file = output_dir / f"{start_time}@dataframe.csv"
            df.to_csv(df_file.resolve())
            config_details_file = output_dir / f"{start_time}@test_config.txt"
            logger.info(config_string(config_details_file.resolve()))
            logger.info(f"Data saved to: {df_file.resolve()}")

    if c.visualisation:

        fig, ax = plt.subplots()
        fig.subplots_adjust(top=0.8)
        ax.set_xlim(0, len(df))
        ax.set_ylim(0, 1)
        ax.set_ylabel("sO2")
        ax.set_xlabel("Sample number")
        x = np.arange(0, len(df))
        y = df["preds"]
        ax.errorbar(
            x, y,
            yerr=df["stdevs"],
            elinewidth=0.1,
            capsize=1,
            ecolor='r',
            errorevery=10,
            markevery=10,
            ls='none',
            ms=1,
            fmt='o'
        )
        ax.plot(x, df["labels"])
        pm = '$\pm$1'
        sampling_type = f"{'random' if c.sample_posterior else pm} sampling"
        ax.set_title(f"sO2 predictions of cINN based on \n"
                  f"{sampling_type} from Gaussian latent space")
        fig.show()

        err_fig, err_ax = plt.subplots()
        err_ax.set_xlim(0, 1)
        err_ax.set_title('Absolute prediction error for sO2 values\n'
                         f'{sampling_type}')
        err_ax.set_xlabel('sO2')
        err_ax.set_ylabel('Absolute error (prediction - ground truth)')
        err_ax.scatter(np.arange(0, 1, 1/len(df)), df["errors"], s=1)
        err_fig.show()

        rel_err_fig, rel_err_ax = plt.subplots()
        rel_err_ax.set_xlim(0, 1)
        rel_err_ax.set_title('Relative prediction error for sO2 values\n'
                             f'{sampling_type}')
        rel_err_ax.set_xlabel('sO2')
        rel_err_ax.set_ylabel('Relative error (Upper and lower stdevs)')
        rel_err_ax.scatter(np.arange(0, 1, 1/len(df)), np.add(df["stdevs_lower"], df["stdevs_upper"]), s=1)
        rel_err_fig.show()
