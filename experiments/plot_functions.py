import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import beta as beta_dist

import jax.numpy as jnp

from PyALE import ale

import seaborn as sns

sns.set(palette="muted")

from training_functions import batched_apply, logit, inverse_logit
from constants import FEATURE_NAMES


def ale_matrix(state, x_orig, *, preprocess, model_features, feature_group_idx):
    class FakeModel:
        def __init__(self, state):
            self.state = state

        def predict(self, df):
            return batched_apply(state, df.values, 1024)[:, 0]

    feature_ranges = np.quantile(
        preprocess.inverse_transform(x_orig), [1e-4, 1 - 1e-4], axis=0
    )
    x_df = pd.DataFrame(np.asarray(x_orig), columns=model_features)

    for grp in feature_group_idx:

        def get_name(feat):
            return FEATURE_NAMES.get(feat, feat)

        nfeat = len(grp)

        fig, axgrid = plt.subplots(nfeat, nfeat, figsize=(3 * nfeat, 3 * nfeat))
        axgrid = np.atleast_2d(axgrid)

        for ax_i in range(nfeat):
            for ax_j in range(nfeat):
                ax = axgrid[ax_i][ax_j]

                if ax_i < ax_j:
                    ax.remove()
                    continue

                feat_i, feat_j = grp[ax_i], grp[ax_j]

                if ax_i == ax_j:
                    feat_name = get_name(model_features[feat_i])

                    ale_df = ale(
                        X=x_df,
                        model=FakeModel(state),
                        feature=[model_features[feat_i]],
                        include_CI=False,
                        feature_type="continuous",
                        plot=False,
                        grid_size=20,
                    )

                    fake_grid = np.zeros((len(ale_df.index), len(model_features)))
                    fake_grid[:, feat_i] = ale_df.index
                    feat_i_inv = preprocess.inverse_transform(fake_grid)[:, feat_i]

                    ax.plot(feat_i_inv, ale_df["eff"].values)
                    ax.set_xlim(*feature_ranges[:, feat_i])
                    ax.set_ylim(-1, 1)
                    ax.set_xlabel(feat_name)
                    ax.set_ylabel(" ")
                    continue

                feat_name_x = get_name(model_features[feat_i])
                feat_name_y = get_name(model_features[feat_j])

                ale_df = ale(
                    X=x_df,
                    model=FakeModel(state),
                    feature=[model_features[feat_i], model_features[feat_j]],
                    include_CI=False,
                    feature_type="continuous",
                    plot=False,
                    grid_size=20,
                )

                fake_grid = np.zeros((len(ale_df.index), len(model_features)))
                fake_grid[:, feat_i] = ale_df.index
                feat_i_inv = preprocess.inverse_transform(fake_grid)[:, feat_i]

                fake_grid = np.zeros((len(ale_df.columns), len(model_features)))
                fake_grid[:, feat_j] = ale_df.columns
                feat_j_inv = preprocess.inverse_transform(fake_grid)[:, feat_j]

                ax.grid(False)
                c = ax.contourf(
                    feat_i_inv,
                    feat_j_inv,
                    ale_df.values.T,
                    levels=np.arange(-0.5, 0.51, 0.05),
                    cmap="RdBu_r",
                )
                cs = ax.contour(
                    feat_i_inv,
                    feat_j_inv,
                    ale_df.values.T,
                    levels=np.arange(-0.5, 0.51, 0.05),
                    colors="black",
                )
                ax.clabel(cs, cs.levels, inline=True, fontsize=7)

                ax.set_xlim(*feature_ranges[:, feat_i])
                ax.set_ylim(*feature_ranges[:, feat_j])
                ax.set_xlabel(feat_name_x)
                ax.set_ylabel(feat_name_y)

        fig.tight_layout(w_pad=2, h_pad=2)


def check_calibration(swag_samples, y):
    plt.figure()

    pred_val = jnp.mean(swag_samples, axis=1)
    pred_std = jnp.std(swag_samples, axis=1)

    bins = inverse_logit(np.arange(pred_val.min(), pred_val.max(), 0.1))
    bin_idx = np.digitize(pred_val, logit(bins))

    calibration_score = 0.0
    norm = 0.0

    for i in range(len(bins)):
        bin_mask = bin_idx == i
        bin_means = pred_val[bin_mask]
        bin_stds = pred_std[bin_mask]

        mean_val = bin_means.mean()
        mean_std = bin_stds.mean()

        mean_val_p = inverse_logit(mean_val)

        true_labels = y[bin_mask]
        if true_labels.sum() == 0:
            continue

        observed_dist = beta_dist(true_labels.sum(), (1 - true_labels).sum())
        observed_mean = observed_dist.mean()
        observed_ci = observed_dist.ppf([0.33, 0.66])

        x_errors = np.abs(np.array(observed_ci) - observed_mean).reshape(2, 1)

        y_errors = np.array(
            [
                mean_val_p - inverse_logit(mean_val - 3 * mean_std),
                inverse_logit(mean_val + 3 * mean_std) - mean_val_p,
            ]
        ).reshape(2, 1)

        plt.errorbar(
            observed_mean, mean_val_p, fmt=".", xerr=x_errors, yerr=y_errors, c="C0"
        )

        weight = 1 / (np.log(observed_ci[1]) - np.log(observed_ci[0]))
        calibration_score += weight * (logit(observed_mean) - mean_val) ** 2
        norm += weight

    plt.plot(bins, bins, "--", color="k", alpha=0.4)
    plt.xlabel("observed")
    plt.ylabel("predicted")

    plt.xscale("log")
    plt.yscale("log")

    return {
        "calibration_err": np.sqrt(calibration_score / norm),
        "pred_range": np.quantile(inverse_logit(pred_val), [1e-5, 1 - 1e-5]).tolist(),
    }
