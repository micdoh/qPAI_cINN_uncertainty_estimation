import torch
import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import qPAI_cINN_uncertainty_estimation.config as c
from qPAI_cINN_uncertainty_estimation.model import WrappedModel, save, load
from qPAI_cINN_uncertainty_estimation.data import prepare_dataloader
from qPAI_cINN_uncertainty_estimation.init_log import init_logger
from qPAI_cINN_uncertainty_estimation.monitoring import config_string


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
        df_file = output_dir / f"{c.load_eval_data_date}@dataframe.csv"
        df = pd.read_csv(df_file.resolve())

    else:

        labels = np.array([])
        preds = np.array([])
        errors = np.array([])
        stdevs_lower = np.array([])
        stdevs_upper = np.array([])

        model.eval()  # Optional when not using Model Specific layer
        for i_val, (data, label) in enumerate(test_dataloader):
            # Transfer Data to GPU if available
            data, label = data.to(c.device), label.to(c.device)

            if c.sample_posterior:

                z_samples = [[] for _ in range(label.shape[0])]
                for i_sample in range(c.n_samples):
                    print(i_sample)
                    # Define input for reverse pass
                    rev_inputs = torch.randn_like(label)
                    # Forward Pass
                    z, _ = model.reverse_sample(data, rev_inputs)
                    z = z.mean(dim=1)  # Take mean of the two features/estimates
                    for i_pred, val in enumerate(z.detach().cpu().numpy()):
                        z_samples[i_pred].append(val)

                z_pred = np.array([np.mean(samples) for samples in z_samples])
                # TODO - is this the correct way to get the standard dev?
                stdev_lower = stdev_upper = np.array([np.std(samples) for samples in z_samples])

            else:
                # Sample with tensors (0,0) for mean, (-1,-1) for lower stdev, (1, 1) for upper stdev
                mean_rev_inputs = torch.zeros_like(label)
                z_pred, _ = model.reverse_sample(data, mean_rev_inputs)

                stdev_lower_rev_inputs = torch.tensor([-1, -1]).repeat(label.shape[0], 1).float().to(c.device)
                z_stdev_lower, _ = model.reverse_sample(data, stdev_lower_rev_inputs)

                stdev_upper_rev_inputs = torch.ones_like(label)
                z_stdev_upper, _ = model.reverse_sample(data, stdev_upper_rev_inputs)

                z_pred = z_pred.mean(dim=1).detach().numpy()
                z_stdev_lower = z_stdev_lower.mean(dim=1).detach().numpy()
                z_stdev_upper = z_stdev_upper.mean(dim=1).detach().numpy()
                stdev_lower = np.subtract(z_pred, z_stdev_lower)
                stdev_upper = np.subtract(z_stdev_upper, z_pred)

            label = label.mean(dim=1).detach().cpu().numpy()

            err = np.subtract(z_pred, label)
            errors = np.append(errors, err)
            labels = np.append(labels, label)
            preds = np.append(preds, z_pred)
            stdevs_lower = np.append(stdevs_lower, stdev_lower)
            stdevs_upper = np.append(stdevs_upper, stdev_upper)
            print(i_val)

        #errors = np.mean(errors[:28800].reshape(-1, 100), axis=1)
        #labels = np.mean(labels[:28800].reshape(-1, 100), axis=1)
        #preds = np.mean(preds[:28800].reshape(-1, 100), axis=1)
        #stdevs_lower = np.mean(stdevs_lower[:28800].reshape(-1, 100), axis=1)
        #stdevs_upper = np.mean(stdevs_upper[:28800].reshape(-1, 100), axis=1)

        df = pd.DataFrame({"errors": errors,
                           "labels": labels,
                           "preds": preds,
                           "stdevs_lower": stdevs_lower,
                           "stdevs_upper": stdevs_upper,
                           })
        df.sort_values(by=["labels"], ascending=False, inplace=True)

        if c.save_eval_data:
            output_dir.mkdir(parents=True, exist_ok=True)
            df_file = output_dir / f"{start_time}@dataframe.csv"
            df.to_csv(df_file.resolve())
            config_details_file = output_dir / f"{start_time}@test_config.txt"
            logger.info(config_string(config_details_file.resolve()))

    if c.visualisation:

        stdevs = np.vstack((df["stdevs_lower"], df["stdevs_upper"]))
        fig, ax = plt.subplots()
        fig.subplots_adjust(top=0.8)
        ax.set_xlim(0, len(df))
        ax.set_ylim(0, 1)
        ax.set_ylabel("sO2")
        ax.set_xlabel("Sample number")
        x = np.arange(0, len(preds))
        y = df["preds"]
        ax.errorbar(
            x, y,
            yerr=stdevs,
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
        err_ax.scatter(np.arange(0, 1, 1/len(df)), errors, ms=1)
        err_fig.show()

        rel_err_fig, rel_err_ax = plt.subplots()
        rel_err_ax.set_xlim(0, 1)
        rel_err_ax.set_title('Relative prediction error for sO2 values\n'
                             f'{sampling_type}')
        rel_err_ax.set_xlabel('sO2')
        rel_err_ax.set_ylabel('Relative error (Upper and lower stdevs)')
        rel_err_ax.scatter(np.arange(0, 1, 1/len(df)), np.add(df["stdevs_lower"], df["stdevs_upper"]), ms=1)
        rel_err_fig.show()
