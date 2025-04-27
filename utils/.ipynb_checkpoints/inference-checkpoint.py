import os
import sys
sys.path.insert(1, '../')
import numpy as np
import pdb
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.stats import norm
import xgboost as xgb
from ppi_py.utils import bootstrap


def train_sampling_rule(X, Y, eta=0.001, max_depth=3, objective='reg:squarederror', boost_rounds=2000):
    # Convert to arrays
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    # Mask for valid labels
    valid_mask = np.isfinite(Y)  # filters out NaN and inf

    # Filter X and Y
    X_clean = X[valid_mask]
    Y_clean = Y[valid_mask]

    dtrain = xgb.DMatrix(X_clean, label=Y_clean, missing=np.nan)
    tree = xgb.train({'eta': eta, 'max_depth': max_depth, 'objective': objective}, dtrain, boost_rounds)
    return tree


def sampling_rule_predict(tree, X):
    return tree.predict(xgb.DMatrix(X))


def confidence_driven_inference(
    estimator,
    Y,
    Yhat,
    sampling_probs,
    sampling_decisions,
    X=None,
    lam=None,
    n_resamples=1000,
    n_resamples_lam=50,
    alpha=0.1,
    alternative="two-sided",
    method="percentile",
):


    weights_lab = sampling_decisions / sampling_probs
    weights_unlab = (1-sampling_decisions) / (1 - sampling_probs)

    if X is None:

        def lam_statistic(Y, Yhat, weights_lab, weights_unlab, estimator=None):
            return {
                "Y": estimator(Y, weights_lab),
                "Yhat": estimator(Yhat, weights_lab),
                "Yhat_unlabeled": estimator(Yhat, weights_unlab),
            }

        if lam is None:
            estimator_dicts = bootstrap(
                [Y, Yhat, weights_lab, weights_unlab],
                lam_statistic,
                n_resamples=n_resamples_lam,
                paired="all",
                statistic_kwargs={"estimator": estimator},
            )
            Y_samples = np.stack(
                [est_dict["Y"] for est_dict in estimator_dicts], axis=0
            )
            Yhat_samples = np.stack(
                [est_dict["Yhat"] for est_dict in estimator_dicts], axis=0
            )
            Yhat_unlabeled_samples = np.stack(
                [est_dict["Yhat_unlabeled"] for est_dict in estimator_dicts],
                axis=0,
            )

            cov_Y_Yhat = (
                np.sum(
                    [
                        np.cov(Y_samples[:, j], Yhat_samples[:, j])[0, 1]
                        for j in range(Y_samples.shape[1])
                    ]
                )
                if len(Y_samples.shape) > 1
                else np.cov(Y_samples, Yhat_samples)[0, 1]
            )
            var_Yhat = (
                np.sum(
                    [
                        np.var(Yhat_samples[:, j])
                        for j in range(Yhat_samples.shape[1])
                    ]
                )
                if len(Yhat_samples.shape) > 1
                else np.var(Yhat_samples)
            )
            var_Yhat_unlabeled = (
                np.sum(
                    [
                        np.var(Yhat_unlabeled_samples[:, j])
                        for j in range(Yhat_unlabeled_samples.shape[1])
                    ]
                )
                if len(Yhat_unlabeled_samples.shape) > 1
                else np.var(Yhat_unlabeled_samples)
            )
            lam = cov_Y_Yhat / (var_Yhat + var_Yhat_unlabeled)

        def rectified_estimator(Y, Yhat, weights_lab, weights_unlab, lam=None):
            return (
                lam * estimator(Yhat, weights_unlab)
                + estimator(Y, weights_lab)
                - lam * estimator(Yhat, weights_lab)
            )
        ppi_pointest = rectified_estimator(Y, Yhat, weights_lab, weights_unlab, lam=lam)

        ppi_bootstrap_distribution = np.array(
            bootstrap(
                [Y, Yhat, weights_lab, weights_unlab],
                rectified_estimator,
                n_resamples=n_resamples,
                paired="all",
                statistic_kwargs={"lam": lam},
            )
        )

    else:

        def lam_statistic(
            X, Y, Yhat, weights_lab, weights_unlab, estimator=None
        ):
            return {
                "XY": estimator(X, Y, weights_lab),
                "XYhat": estimator(X, Yhat, weights_lab),
                "XYhat_unlabeled": estimator(X, Yhat, weights_unlab),
            }

        if lam is None:
            estimator_dicts = bootstrap(
                [X, Y, Yhat, weights_lab, weights_unlab],
                lam_statistic,
                n_resamples=n_resamples_lam,
                paired="all",
                statistic_kwargs={"estimator": estimator},
            )
            XY_samples = np.stack(
                [est_dict["XY"] for est_dict in estimator_dicts], axis=0
            )
            XYhat_samples = np.stack(
                [est_dict["XYhat"] for est_dict in estimator_dicts], axis=0
            )
            XYhat_unlabeled_samples = np.stack(
                [est_dict["XYhat_unlabeled"] for est_dict in estimator_dicts],
                axis=0,
            )

            cov_XY_XYhat = (
                np.sum(
                    [
                        np.cov(XY_samples[:, j], XYhat_samples[:, j])[0, 1]
                        for j in range(XY_samples.shape[1])
                    ]
                )
                if len(XY_samples.shape) > 1
                else np.cov(XY_samples, XYhat_samples)[0, 1]
            )
            var_XYhat = (
                np.sum(
                    [
                        np.var(XYhat_samples[:, j])
                        for j in range(XYhat_samples.shape[1])
                    ]
                )
                if len(XYhat_samples.shape) > 1
                else np.var(XYhat_samples)
            )
            var_XYhat_unlabeled = (
                np.sum(
                    [
                        np.var(XYhat_unlabeled_samples[:, j])
                        for j in range(XYhat_unlabeled_samples.shape[1])
                    ]
                )
                if len(XYhat_unlabeled_samples.shape) > 1
                else np.var(XYhat_unlabeled_samples)
            )

            lam = cov_XY_XYhat / (var_XYhat + var_XYhat_unlabeled)

        def rectified_estimator(
            X, Y, Yhat, weights_lab, weights_unlab, lam=None
        ):
            return (
                lam * estimator(X, Yhat, weights_unlab)
                + estimator(X, Y, weights_lab)
                - lam * estimator(X, Yhat, weights_lab)
            )

        ppi_pointest = rectified_estimator(
            X, Y, Yhat, weights_lab, weights_unlab, lam=lam
        )

        ppi_bootstrap_distribution = np.array(
            bootstrap(
                [X, Y, Yhat, weights_lab, weights_unlab],
                rectified_estimator,
                n_resamples=n_resamples,
                paired="all",
                statistic_kwargs={"lam": lam},
            )
        )

    # Deal with the different types of alternative hypotheses
    if alternative == "two-sided":
        alpha_lower = alpha / 2
        alpha_upper = alpha / 2
    elif alternative == "larger":
        alpha_lower = alpha
        alpha_upper = 0
    elif alternative == "smaller":
        alpha_lower = 0
        alpha_upper = alpha

    # Compute the lower and upper bounds depending on the method
    if method == "percentile":
        lower_bound = np.quantile(
            ppi_bootstrap_distribution, alpha_lower, axis=0
        )
        upper_bound = np.quantile(
            ppi_bootstrap_distribution, 1 - alpha_upper, axis=0
        )
    elif method == "basic":
        lower_bound = 2 * ppi_pointest - np.quantile(
            ppi_bootstrap_distribution, 1 - alpha_lower, axis=0
        )
        upper_bound = 2 * ppi_pointest - np.quantile(
            ppi_bootstrap_distribution, alpha_upper, axis=0
        )
    else:
        raise ValueError(
            "Method must be either 'percentile' or 'basic'. The others are not implemented yet... want to contribute? ;)"
        )

    if alternative == "two-sided":
        return lower_bound, upper_bound
    elif alternative == "larger":
        return -np.inf, upper_bound
    elif alternative == "smaller":
        return lower_bound, np.inf
    else:
        raise ValueError(
            "Alternative must be either 'two-sided', 'larger' or 'smaller'."
        )