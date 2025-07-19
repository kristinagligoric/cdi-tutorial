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
from utils.qualtrics import create_and_activate_surveys
from utils.prolific import run_prolific_annotation_pipeline
from utils.mturk import run_mturk_annotation_pipeline
from scipy.stats import norm, bernoulli

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

    weights_lab = np.divide(
    sampling_decisions,
    sampling_probs,
    out=np.zeros_like(sampling_decisions, dtype=float),
    where=sampling_probs != 0
    )    

    weights_unlab = np.divide(
    1-sampling_decisions,
    1-sampling_probs,
    out=np.zeros_like(sampling_decisions, dtype=float),
    where=(1 - sampling_probs) != 0
    )

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
        cdi_pointest = rectified_estimator(Y, Yhat, weights_lab, weights_unlab, lam=lam)

        cdi_bootstrap_distribution = np.array(
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

        cdi_pointest = rectified_estimator(
            X, Y, Yhat, weights_lab, weights_unlab, lam=lam
        )

        cdi_bootstrap_distribution = np.array(
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
            cdi_bootstrap_distribution, alpha_lower, axis=0
        )
        upper_bound = np.quantile(
            cdi_bootstrap_distribution, 1 - alpha_upper, axis=0
        )
    elif method == "basic":
        lower_bound = 2 * cdi_pointest - np.quantile(
            cdi_bootstrap_distribution, 1 - alpha_lower, axis=0
        )
        upper_bound = 2 * cdi_pointest - np.quantile(
            cdi_bootstrap_distribution, alpha_upper, axis=0
        )
    else:
        raise ValueError(
            "Method must be either 'percentile' or 'basic'. The others are not implemented yet... want to contribute? ;)"
        )

    if alternative == "two-sided":
        return cdi_pointest, (lower_bound, upper_bound)
    elif alternative == "larger":
        return cdi_pointest, (-np.inf, upper_bound)
    elif alternative == "smaller":
        return cdi_pointest, (lower_bound, np.inf)
    else:
        raise ValueError(
            "Alternative must be either 'two-sided', 'larger' or 'smaller'."
        )


def collect_initial_human_annotations(data, df, COLLECT_HUMAN, HUMAN_SOURCE, burnin_steps, N, random_state, human_annotation_parameters):
    categories = human_annotation_parameters["categories"]
    annotation_instruction = human_annotation_parameters["annotation_instruction"]
    annotation_instructions = human_annotation_parameters["annotation_instructions"]
    QUALTRICS_API_URL = human_annotation_parameters["QUALTRICS_API_URL"]
    QUALTRICS_API_KEY = human_annotation_parameters["QUALTRICS_API_KEY"]
    task_title = human_annotation_parameters["task_title"]
    task_description = human_annotation_parameters["task_description"]
    reward = human_annotation_parameters["reward"]
    estimated_time = human_annotation_parameters["estimated_time"]
    maximum_allowed_time = human_annotation_parameters["maximum_allowed_time"]
    HEADERS = human_annotation_parameters["HEADERS"]
    BASE_URL = human_annotation_parameters["BASE_URL"]
    BATCH_TIMEOUT = human_annotation_parameters["BATCH_TIMEOUT"]
    task_reward = human_annotation_parameters["task_reward"]
    minimum_approval_rate = human_annotation_parameters["minimum_approval_rate"]
    minimum_tasks_approved = human_annotation_parameters["minimum_tasks_approved"]
    AWS_ACCESS_KEY_ID = human_annotation_parameters["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = human_annotation_parameters["AWS_SECRET_ACCESS_KEY"]
    positive_class = human_annotation_parameters["positive_class"]

    if COLLECT_HUMAN:
        texts_to_annotate = list(data.loc[:burnin_steps-1, 'text'].values)

        if HUMAN_SOURCE == "Prolific":
            # create Qualtrics annotation interface and get annotation task URLs
            survey_links = create_and_activate_surveys(
                texts_to_annotate=texts_to_annotate,
                categories=categories,
                annotation_instruction=annotation_instruction,
                QUALTRICS_API_URL=QUALTRICS_API_URL,
                QUALTRICS_API_KEY=QUALTRICS_API_KEY)

            # run the Prolific annotation pipeline
            annotations = run_prolific_annotation_pipeline(
                survey_links=list(survey_links.values()),
                name_prefix=task_title,
                description=task_description,
                reward=reward,
                estimated_time=estimated_time,
                max_time=maximum_allowed_time,
                HEADERS=HEADERS,
                BASE_URL=BASE_URL,
                QUALTRICS_API_URL=QUALTRICS_API_URL,
                QUALTRICS_API_KEY=QUALTRICS_API_KEY,
                BATCH_TIMEOUT=BATCH_TIMEOUT
            )

        if HUMAN_SOURCE == "MTURK":
            annotations = run_mturk_annotation_pipeline(pd.DataFrame(texts_to_annotate, columns=['Text']),
                                                        annotation_instructions=annotation_instructions,
                                                        task_title=task_title,
                                                        task_description=task_description,
                                                        task_reward=task_reward,
                                                        minimum_approval_rate=minimum_approval_rate,
                                                        minimum_tasks_approved=minimum_tasks_approved,
                                                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                                                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

        data.loc[:burnin_steps-1, 'human'] = pd.Series(annotations).apply(lambda x: 1 if x.lower() == positive_class.lower() else 0).values

    else:
        # load the existing annotations we already collected
        df['Prediction_human'] = pd.read_csv('data/politeness_dataset.csv').sample(n=N, random_state=random_state)['Prediction_human'].values
        df = df['Prediction_human'].reset_index()
        # initialize the first burnin_steps annotations
        data.loc[:burnin_steps-1, 'human'] = df['Prediction_human'].values[:burnin_steps]

    return data


def run_adaptive_sampling(data, df, n, burnin_steps, n_human, n_batches, tau,
                          COLLECT_HUMAN, HUMAN_SOURCE, human_annotation_parameters):

    categories = human_annotation_parameters["categories"]
    annotation_instruction = human_annotation_parameters["annotation_instruction"]
    annotation_instructions = human_annotation_parameters["annotation_instructions"]
    QUALTRICS_API_URL = human_annotation_parameters["QUALTRICS_API_URL"]
    QUALTRICS_API_KEY = human_annotation_parameters["QUALTRICS_API_KEY"]
    task_title = human_annotation_parameters["task_title"]
    task_description = human_annotation_parameters["task_description"]
    reward = human_annotation_parameters["reward"]
    estimated_time = human_annotation_parameters["estimated_time"]
    maximum_allowed_time = human_annotation_parameters["maximum_allowed_time"]
    HEADERS = human_annotation_parameters["HEADERS"]
    BASE_URL = human_annotation_parameters["BASE_URL"]
    BATCH_TIMEOUT = human_annotation_parameters["BATCH_TIMEOUT"]
    task_reward = human_annotation_parameters["task_reward"]
    minimum_approval_rate = human_annotation_parameters["minimum_approval_rate"]
    minimum_tasks_approved = human_annotation_parameters["minimum_tasks_approved"]
    AWS_ACCESS_KEY_ID = human_annotation_parameters["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = human_annotation_parameters["AWS_SECRET_ACCESS_KEY"]
    positive_class = human_annotation_parameters["positive_class"]

    confidence = data['llm_conf'].to_numpy().reshape((n,1))
    confidence_burnin = confidence[:burnin_steps]
    H = data['human'].to_numpy()
    H_burnin = H[:burnin_steps]
    Hhat = data['llm'].to_numpy()
    Hhat_burnin = Hhat[:burnin_steps]
    SP = np.zeros(n)
    SD = np.zeros(n)
    SP[:burnin_steps] = np.ones(burnin_steps)
    SD[:burnin_steps] = np.ones(burnin_steps)

    sampling_rule = train_sampling_rule(confidence_burnin, (H_burnin - Hhat_burnin)**2)
    sampling_probs_unnormed = np.clip(np.sqrt(sampling_rule_predict(sampling_rule, confidence)), 1e-4, 1)
    avg_sampling_probs = np.mean(sampling_probs_unnormed)
    frac_human_adjusted = (n_human - burnin_steps) / (n - burnin_steps)

    batch_size = (n - burnin_steps) // n_batches
    for b in range(n_batches):
        if b < (n_batches - 1):
            batch_inds = np.array(range(burnin_steps + b * batch_size, burnin_steps + (b + 1) * batch_size))
        else:
            batch_inds = np.array(range(burnin_steps + b * batch_size, n))

        sampling_probs = sampling_probs_unnormed[batch_inds] / avg_sampling_probs * frac_human_adjusted
        sampling_probs = np.clip((1 - tau) * sampling_probs + tau * frac_human_adjusted, 0, 1)

        if np.isnan(sampling_probs).all():
            print(f"Training the model failed at batch {b+1}/{n_batches}... Quitting.")
            break

        labeling_decisions = bernoulli.rvs(sampling_probs)
        indices_to_label = batch_inds[np.where(labeling_decisions)]

        print()
        print(f"Collecting batch {b+1}/{n_batches}...")

        if COLLECT_HUMAN:
            texts_to_annotate = list(data.loc[indices_to_label, 'text'].values)

            if HUMAN_SOURCE == "Prolific":
                survey_links = create_and_activate_surveys(
                    texts_to_annotate=texts_to_annotate,
                    categories=categories,
                    annotation_instruction=annotation_instruction,
                    QUALTRICS_API_URL=QUALTRICS_API_URL,
                    QUALTRICS_API_KEY=QUALTRICS_API_KEY)

                annotations = run_prolific_annotation_pipeline(
                    survey_links=list(survey_links.values()),
                    name_prefix=task_title,
                    description=task_description,
                    reward=reward,
                    estimated_time=estimated_time,
                    max_time=maximum_allowed_time,
                    HEADERS=HEADERS,
                    BASE_URL=BASE_URL,
                    QUALTRICS_API_URL=QUALTRICS_API_URL,
                    QUALTRICS_API_KEY=QUALTRICS_API_KEY,
                    BATCH_TIMEOUT=BATCH_TIMEOUT
                )

            elif HUMAN_SOURCE == "MTURK":
                annotations = run_mturk_annotation_pipeline(
                    pd.DataFrame(texts_to_annotate, columns=['Text']),
                    annotation_instructions=annotation_instructions,
                    task_title=task_title,
                    task_description=task_description,
                    task_reward=task_reward,
                    minimum_approval_rate=minimum_approval_rate,
                    minimum_tasks_approved=minimum_tasks_approved,
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
                )

            H[indices_to_label] = pd.Series(annotations).apply(lambda x: 1 if x.lower() == positive_class.lower() else 0).values

        else:
            H[indices_to_label] = df['Prediction_human'].iloc[list(indices_to_label)]
            print(f"Collecting {len(df['Prediction_human'].iloc[list(indices_to_label)])} human annotations.")

        collected_inds = np.where(labeling_decisions)

        SP[batch_inds] = sampling_probs
        SD[batch_inds] = labeling_decisions

        if b < (n_batches - 1):
            sampling_rule = train_sampling_rule(
                confidence[collected_inds],
                (H[collected_inds] - Hhat[collected_inds])**2
            )
            sampling_probs_unnormed = np.clip(np.sqrt(sampling_rule_predict(sampling_rule, confidence)), 1e-4, 1)
            avg_sampling_probs = np.mean(sampling_probs_unnormed)

    data.loc[list(collected_inds[0]), 'human'] = H[list(collected_inds)][0]
    data['sampling_probs'] = SP
    data['sampling_decisions'] = SD
    print(f"{len(data.dropna(subset=['human']))} human datapoints collected in total.")
    
    return data