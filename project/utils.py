"""A module containing utility functions used throughout the project.

This module contains functions that is imported and used throughout this project. The functions are used for:
- preprocessing
- visualisation
- cross validation
- hyperparameter tuning

Author: Socretquuliqaa <lee.socret@gmail.com>
Created: 09/04/2023
"""

import os
import csv
import shelve
import numpy as np
import pandas as pd
import seaborn as sns
from skopt import BayesSearchCV  # pip install scikit-optimize
from tqdm.auto import tqdm
from functools import reduce
import matplotlib.pyplot as plt
from itertools import islice, chain, repeat, combinations, product
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from skopt.plots import plot_objective
from sklearn import metrics
import timeit



def read_raw_measurement_file(filepath, columns=None):
    """ Read raw measurement file.

    Args:
    ---
    - `filepath`: str
      Path to the raw measurement file.
    - `columns`: List[str], optional
      The list of column names. If it is None, the column names default to the index of the column. (Default is None)

    Returns:
    ---
    Tuple[DataFrame, int, int, int]
      A tuple containing the DataFrame, start time, end time, sampling rate
    """
    info_df = pd.read_csv(filepath, header=None, nrows=2)
    start_time, sampling_rate = info_df.iloc[0, 0], int(info_df.iloc[1, 0])

    df = pd.read_csv(
        filepath,
        header=1,
        names=columns if columns is not None else list(
            range(len(info_df.iloc[0])))
    )

    df['timestamp'] = df.apply(
        lambda x: start_time + (x.name / sampling_rate), axis=1)

    return df, start_time, start_time + (len(df) // sampling_rate), sampling_rate


def read_ibi_measurement_file(filepath):
    """ Read IBI measurement file.

    Args:
    ---
    - `filepath`: str
      Path to the raw measurement file.

    Returns:
    ---
    Tuple[DataFrame, int, int]
      A tuple containing the DataFrame, start time, end time
    """
    info_df = pd.read_csv(filepath, header=None, nrows=2)
    start_time = info_df.iloc[0, 0]

    df = pd.read_csv(
        filepath,
        header=0,
        names=['t', 'inter_beat_interval']
    )

    df['timestamp'] = df.apply(lambda x: start_time + x['t'], axis=1)

    return df.drop('t', axis=1), start_time, start_time + df.iloc[-1][0]


def get_experiment_start_end_time(participant):
    """ Get the latest start time and earliest end time of the `participant` experiment.

    Args:
    ---
    - `participant`: str
      Path to the directory containing the raw data of the participant.

    Returns:
    ---
    Tuple[int, int]
      The latest start time, The earliest end time
    """
    t = []

    # ACC.csv, TEMP.csv, HR.csv, BVP.csv, EDA.csv
    for measurement in ["ACC.csv", "TEMP.csv", "HR.csv", "BVP.csv", "EDA.csv"]:
        df, start_time, end_time, _ = read_raw_measurement_file(
            os.path.join(participant, measurement))
        t.append({
            "measurement": measurement,
            "start_time": start_time,
            "end_time": end_time,
        })

    # IBI.csv
    df, start_time, end_time = read_ibi_measurement_file(
        os.path.join(participant, "IBI.csv"))
    t.append({
        "measurement": "IBI.csv",
        "start_time": start_time,
        "end_time": end_time
    })

    t = pd.DataFrame(t)

    return t["start_time"].astype(int).max(), t["end_time"].astype(int).min()


def adjust_measurement_sampling_rate(measurement, start_time, end_time, sampling_rate):
    """ Adjust the sampling rate of a measurement.

    Args:
    ---
    - `measurement`: DataFrame
      The measurement DataFrame.
    - `start_time`: int
      Start time in UNIX timestamp format.
    - `end_time`: int
      End time in UNIX timestamp format.
    - `sampling_rate`: int
      The sampling rate to adjust to.

    Returns:
    ---
    DataFrame
      The adjusted measurement DataFrame.
    """
    num_seconds = int(end_time - start_time) + \
        1  # The number of seconds 0->1 = 2 seconds

    steps = np.arange(0, 1, step=1/sampling_rate)
    seconds = np.arange(start_time, end_time+1, step=1)

    timestamps = pd.DataFrame({"timestamp": np.tile(
        steps, (num_seconds)) + np.repeat(seconds, repeats=sampling_rate, axis=0)})

    # keep all timestamps, if row in measurement does not have corresponding timestamp, then drop
    return pd.merge(timestamps, measurement, on=['timestamp'], how='left')


def read_participant_task_tags(filepath):
    """ Read the tag file of a participant by grouping them into pairs.

    Args:
    ---
    - `filepath`: str
      Path to tag file of the participant.

    Return:
    ---
    List[Tuple[float, float]]
      The list of start and end time grouped as tuple pair.
      If number of elements is odd, the final pair is padded with None.
    """
    df = [float(i[0]) for i in pd.read_csv(
        filepath, header=None).values.tolist()]
    df = chain(iter(df), repeat(None))
    return list(iter(lambda: tuple(islice(df, 2)), (None,) * 2))


def determine_timestamp_stress_status_of_participant(measurements, task_tags):
    """ Label each timestamp as stress or non-stress.

    Args:
    ---
    - `measurements`: DataFrame
      The combined measurements of a participant.
    - `task_tags`: List[Tuple[float, float]]
      The list of start and end time grouped as tuple pairs.

    Returns:
    ---
    DataFrame
      A new DataFrame with `stress` column.
    """
    temp = measurements.copy()

    def duration_stress_task(x):
        in_test = False

        # only select the first three tasks because they are the stress inducing task
        for (start_time, end_time) in task_tags[:3]:
            if start_time <= x <= end_time:
                in_test = True
                break

        return in_test

    temp['stress'] = temp['timestamp'].apply(lambda x: duration_stress_task(x))
    return temp


def combine_participant_measurements(participant, sampling_rate=1):
    """ Combine the measurements of `participant` into one dataframe with consistent `sampling_rate`.

    Args:
    ---
    - `participant`: str
      Path to the directory of the `participant`'s raw measurements.
    - `sampling_rate`: int, optional
      The sampling rate to adjust to. (Default is 1)

    Returns:
    ---
    Tuple[DataFrame, int, int]
      A tuple containing, the combined dataframe, start time, and end time of `participant` experiment.
    """
    column_mappings = {
        "ACC.csv": ["accel_x", "accel_y", "accel_z"],
        "TEMP.csv": ["skin_temp"],
        "HR.csv": ["heart_rate"],
        "BVP.csv": ["blood_volume_pulse"],
        "EDA.csv": ["eda"],
    }

    measurements = []

    start_time, end_time = get_experiment_start_end_time(participant)

    for measurement in ["ACC.csv", "TEMP.csv", "HR.csv", "BVP.csv", "EDA.csv"]:
        df, _, _, _ = read_raw_measurement_file(
            os.path.join(participant, measurement),
            columns=column_mappings[measurement]
        )
        df = adjust_measurement_sampling_rate(
            df, start_time, end_time, sampling_rate)
        # handle na values generated by upsampling
        df = df.fillna(method='ffill')
        measurements.append(df)

    df, _, _ = read_ibi_measurement_file(os.path.join(participant, "IBI.csv"))
    df = adjust_measurement_sampling_rate(
        df, start_time, end_time, sampling_rate)
    measurements.append(df)

    combined = reduce(lambda left, right: pd.merge(left, right, on=[
                      'timestamp'], how='left'), measurements)  # keep rows without timestamp

    return combined, start_time, end_time


def read_participant_measurements(participant, sampling_rate=1):
    """ Read a `participant` measurements from their raw data directory.

    Args:
    ---
    - `participant`: str
      The path to the `participant` raw data directory.
    - `sampling_rate`: int, optional
      The sampling rate to adjust the measurements to. (Default is 1)

    Returns:
    ---
    Tuple[DataFrame, int, int]
      A tuple containing the `participant` combined and tagged DataFrame, start time, and end time of their experiment.
    """
    combined, start_time, end_time = combine_participant_measurements(participant, sampling_rate)
    task_tags = read_participant_task_tags(os.path.join(participant, f"tags_{os.path.basename(os.path.normpath(participant))}.csv"))
    tagged = determine_timestamp_stress_status_of_participant(combined, task_tags)
    return tagged, start_time, end_time


def plot_experiment_measurement_as_timeseries(df, measurement):
    """ Plot a participant experiment as timeseries.

    Args:
    ---
    - `df`: DataFrame
      The participant's experiment data.
    - `measurement`: str
      The measurement of interests.
    """
    events = df['stress'].diff()
    events = df[events == True]['timestamp'].values.tolist()
    events = [(events[i], events[i+1]) for i in range(0, len(events), 2)]
    events = list(zip([('Stroop CW', 'r'), ('Trier Social Scale Test',
                  'g'), ('Hyperventilation Tests', 'b')], events))

    sns.lineplot(data=df, x='timestamp', y=measurement)

    avg_stress = df[df['stress'] == True][measurement].mean()
    avg_non_stress = df[df['stress'] == False][measurement].mean()

    plt.axhline(
        y=avg_stress, label=f'Avg Stress ({round(avg_stress, 2)})', color='r')
    plt.axhline(y=avg_non_stress,
                label=f'Avg Non Stress ({round(avg_non_stress, 2)})', color='b')

    # draw shades over events
    for event in events:
        (name, color), (start_time, end_time) = event
        duration = int((end_time - start_time) // 60)
        plt.axvspan(start_time, end_time, alpha=0.1, lw=0,
                    label=f"{name} ({duration} mins)", color=color)

    plt.legend(loc='upper right')


def calculate_imbalance_ratio(df):
    """ Calculates the imbalance ratio of `df`.

    Args:
    ---
    - `df`: DataFrame
       The dataframe representing the dataset.

    Returns:
    ---
    Float
      The imbalance ratio.
    """
    return round(len(df[df['stress'] == True]) / len(df[df['stress'] == False]), 2)


def clip_participant_experiment_data(df, leading=240, trailing=240):
    """ Remove the leading and trailing data from an experiment data `df`.

    Args:
    ---
    - `df`: DataFrame
      The participant experiment data.
    - `leading`: int, optional
      The number of seconds before the first stress period. (Default to 240)
    - `trailing`: int, optional
      The number of seconds after the last stress period. (Default to 240)

    Returns:
    ---
    DataFrame
      The clipped experiment data.
    """
    temp = df.copy()
    events = temp['stress'].diff()
    events = temp[events == True]['timestamp'].values.tolist()
    temp = temp[temp['timestamp'] >= events[0] - leading]
    temp = temp[temp['timestamp'] <= events[-1] + trailing]
    return temp


def leave_one_participant_out_cv(df):
    """ A generator for performing leave one participant out cross validation.

    Args:
    ---
    - `df`: DataFrame
      The dataset to perform cv on.

    Returns:
    ---
    Generator[Tuple[List[str], List[str]]]
      A generator that yields a tuple of participants in training set, participant in the validation set.
    """
    participants = df['participant'].unique().tolist()
    for participant in participants:
        yield [p for p in participants if p != participant], participant


def get_all_combinations(options):
    """ Returns all combinations of different sizes from `options`.

    Args:
    ---
    - `options`: List[Any]
      A python list.

    Returns:
    ---
    List[Tuple[Any]]
      A python list containing all the possible combinations of `options`.
    """
    results = []

    for i in range(1, len(options)+1):
        results += list(combinations(options, i))

    return results


def sliding_windows(num_samples, window_size=5, stride=1, future=1):
    """ Returns sliding window idxes of size `window_size` and `stride` for usage with numpy indexing.

    Args:
    ---
    - `num_samples`: int
      The number of samples in the dataset.
    - `window_size`: int
      The size of the sliding window.
    - `stride`: int
      The movement step size of the sliding window.

    Returns:
    ---
    Tuple[ndarray, ndarray]
      A tuple of X_indxes, y_indxes.

    References:
    ---
    - https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
    """
    max_index = num_samples - window_size - future

    sub_windows = (0 + np.expand_dims(np.arange(window_size), 0) + np.expand_dims(np.arange(max_index, step=stride), 0).T)

    return sub_windows


def prepare_data(df, inner_cv_n_split, outer_cv_n_split, params, test=True, df_test=pd.DataFrame()):
    """Prepare data for CV and testing.

    Args:
        df: Input Dataframe containing original timeseries
        label_key: Column name for tested outcome
        models: Dict with ML config
        inner_cv_n_split: Number of folds for inner CV loop
        outer_cv_n_split: Number of folds for outer CV loop
        features: List of signals from which to extract features in DF dataframe.
        test: If True, prepare data for testing model
        df_test: dataframe with the test data (the left-out participants, ids 28--35)

    Returns:
        Dictionary with data ready for training or testing model
    """
    X_train, y_train = df.drop(['stress'], axis=1), df['stress']
    X_train, y_train, groups_train = preprocess(X_train, y_train, features=params['features'],
                                                window_size=params['window_size'],
                                                stride=params['stride'],
                                                future=params['future'] if 'future' in params.keys() else 1,
                                                return_groups=True)

    inner_cv = GroupKFold(n_splits=inner_cv_n_split)
    outer_cv = GroupKFold(n_splits=outer_cv_n_split)

    input_for_cv = {"X": X_train, "y": y_train, "groups": groups_train, "outer_cv": outer_cv, "inner_cv": inner_cv}
    if test:
        X_test, y_test, groups_test = preprocess(df_test.drop(['stress'], axis=1), df_test['stress'],
                                                 features=params['features'], window_size=params['window_size'],
                                                 future=params['future'] if 'future' in params.keys() else 1,
                                                 stride=params['stride'], return_groups=True)
        input_for_cv['X_test'] = X_test
        input_for_cv['y_test'] = y_test
        input_for_cv['groups_test'] = groups_test

    return input_for_cv


def preprocess(X, y=None, window_size=240, stride=32, features='all', return_groups=False, future=1, **kargs):  # this is added for parameter spread support
    """ Preprocess time series using sliding windows of `window_size` and `stride`.

    Args:
    ---
    - `X`: DataFrame
      The features time series dataframe.
    - `y`: DataFrame, optional.
      The target dataframe. (Default is None)
    - `window_size`: int
      The window size of the sliding window.
    - `features`: List[str], optional
      The list of features from EmpaticaE4 to use for feature engineering. (all = ["ACC", "IBI", "TEMP", "EDA", "HR"])
    - `future`: int
      The amount of rows into the future we want to predict (e.g., to see if we can predict stress 1 minute ahead of it occurring)
    Returns:
    ---
    Tuple[ndarray, ndarray]
      A tuple in the form of X, y.
    """
    if features == 'all':
        features = ["ACC", "IBI", "TEMP", "EDA", "HR"]
    columns = X.columns.tolist()
    Xs = X.to_numpy()
    X_idxes = sliding_windows(len(Xs), window_size=window_size, stride=stride, future=future)
    Xs = Xs[X_idxes]
    if y is not None:
        ys = y.to_numpy()
        ys = ys[X_idxes[:, -1]+future]
    if return_groups:
        groups = X['participant'].to_numpy()
        groups = groups[X_idxes[:, -1]+future]
    del X_idxes

    new_columns = []
    if "HR" in features:
        heart_rates = Xs[:, :, columns.index("heart_rate")]
        heart_rates = pd.DataFrame(data=np.concatenate([np.expand_dims(np.mean(heart_rates, axis=-1), axis=-1),
                                                        np.expand_dims(np.std(heart_rates, axis=-1), axis=-1),
                                                        ], axis=-1),
                                   columns=["heart_rate_mean", "heart_rate_std"]
                                   )
        new_columns += [heart_rates]

    if "IBI" in features:
        ibis = Xs[:, :, columns.index("inter_beat_interval")]
        ibis = pd.DataFrame(data=np.sqrt(np.mean(np.power(np.diff(ibis, axis=-1), 2), axis=-1)),
                            # Root Mean Square of Successive Differences between Normal Heartbeats (RMSSD)
                            columns=["hrv"]
                            )
        new_columns += [ibis]

    if "TEMP" in features:
        skin_temp = Xs[:, :, columns.index("skin_temp")]
        skin_temp = pd.DataFrame(data=np.concatenate([np.expand_dims(np.mean(skin_temp, axis=-1), axis=-1),
                                                      np.expand_dims(np.std(skin_temp, axis=-1), axis=-1),
                                                      np.expand_dims(np.max(skin_temp, axis=-1), axis=-1),
                                                      np.expand_dims(np.min(skin_temp, axis=-1), axis=-1),
                                                      ], axis=-1),
                                 columns=["skin_temp_mean", "skin_temp_std", "skin_temp_max", "skin_temp_min"]
                                 )
        new_columns += [skin_temp]

    if "EDA" in features:
        edas = Xs[:, :, columns.index("eda")]
        edas = pd.DataFrame(data=np.concatenate([np.expand_dims(np.mean(edas, axis=-1), axis=-1),
                                                 np.expand_dims(np.std(edas, axis=-1), axis=-1),
                                                 np.expand_dims(np.max(edas, axis=-1), axis=-1),
                                                 np.expand_dims(np.min(edas, axis=-1), axis=-1),
                                                 ], axis=-1),
                            columns=["eda_mean", "eda_std", "eda_max", "eda_min"]
                            )
        new_columns += [edas]

    if "ACC" in features:
        accel_xs = Xs[:, :, columns.index("accel_x")].astype('float')
        accel_xs = pd.DataFrame(data=np.concatenate([np.expand_dims(np.mean(accel_xs, axis=-1), axis=-1),
                                                     np.expand_dims(np.std(accel_xs, axis=-1), axis=-1),
                                                     ], axis=-1),
                                columns=["accel_x_mean", "accel_x_std"]
                                )

        accel_ys = Xs[:, :, columns.index("accel_y")].astype('float')
        accel_ys = pd.DataFrame(data=np.concatenate([np.expand_dims(np.mean(accel_ys, axis=-1), axis=-1),
                                                     np.expand_dims(np.std(accel_ys, axis=-1), axis=-1),
                                                     ], axis=-1),
                                columns=["accel_y_mean", "accel_y_std"]
                                )

        accel_zs = Xs[:, :, columns.index("accel_z")].astype('float')
        accel_zs = pd.DataFrame(data=np.concatenate([np.expand_dims(np.mean(accel_zs, axis=-1), axis=-1),
                                                     np.expand_dims(np.std(accel_zs, axis=-1), axis=-1),
                                                     ], axis=-1),
                                columns=["accel_z_mean", "accel_z_std"]
                                )

        accel_3ds = Xs[:, :, [columns.index("accel_x"), columns.index(
            "accel_y"), columns.index("accel_z")]].astype('float')
        accel_3ds = np.power(accel_3ds, 2)
        accel_3ds = np.sum(accel_3ds, axis=-1)
        accel_3ds = np.sqrt(accel_3ds)
        accel_3ds = pd.DataFrame(data=np.concatenate([np.expand_dims(np.mean(accel_3ds, axis=-1), axis=-1),
                                                      np.expand_dims(np.std(accel_3ds, axis=-1), axis=-1),
                                                      ], axis=-1),
                                 columns=["accel_3d_mean", "accel_3d_std"]
                                 )
        new_columns += [accel_xs, accel_ys, accel_zs, accel_3ds]

    if return_groups:
        return pd.concat(new_columns, axis=1), ys if y is not None else None, groups

    return pd.concat(new_columns, axis=1), ys if y is not None else None



def fit_bayes_cv(X, y, groups, inner_cv, n_bayes_iter, param_grid, scoring, n_jobs, pipeline, nested=True,
                 outer_cv=None, name_optim_file=None, filename_base='Results/Models/Optim/'):
    """Peform Bayesian optimization for a model.

    Args:
        X: Input features
        y: Labels
        groups: For grouped CV
        pipeline: skLearn pipeline
        inner_cv: SkLearn CV Fold object with the inner CV loop
        n_bayes_iter: Number of Bayesian Optimization search iterations
        param_grid: Gridsearch params
        scoring: SkLearn scorer function,
        n_jobs: Number of CPU cores,
        nested: Whether to use Nested CV,
        outer_cv: SkLearn CV Fold object with the outer CV loop (for nested: True)
        name_optim_file: Name of file to store optimisation visualisation
    Returns:
        Dict with Bayesian optimisation validation results
    """
    outer_cv_results = {"auc": [], "precision": [], "recall": [], "roc_curve": [], "best_train": [], "best_val": [],
                        "best_params": [], "best_estimator": [],
                        "prediction_df": []}
    if nested:
        outer_loop = outer_cv.split(X, y, groups)
    else:
        outer_loop = [(0, 0)]

    for i, (train, test) in enumerate(outer_loop):
        print('\t\tRunning iteration {}/{} of the outer loop'.format(i+1, outer_cv.get_n_splits()))
        if nested:
            X_train = X.iloc[train]
            y_train = [y[i] for i in train]
            group_train = [groups[i] for i in train]
            X_test = X.iloc[test]
            y_test = [y[i] for i in test]
            group_test = [groups[i] for i in test]
        else:
            X_train = X
            y_train = y
            group_train = groups
            X_test, y_test = None, None
            group_test = None

        inner_groups = group_train

        # Sklearn cannot handle multi-index
        index_train = X_train.index
        X_train.reset_index(drop=True, inplace=True)
        if nested:
            index_test = X_test.index
            X_test.reset_index(drop=True, inplace=True)

        #   Bayes cv_clf
        start = timeit.default_timer()
        cv_clf = BayesSearchCV(pipeline, param_grid, n_iter=n_bayes_iter, cv=inner_cv, scoring=scoring, n_jobs=n_jobs,
                               n_points=max(1, n_jobs), pre_dispatch=2 * n_jobs, return_train_score=True)
        cv_clf.fit(X_train, y_train, groups=inner_groups)

        duration = timeit.default_timer() - start

        #   Compare train vs. validation vs. test scores (training curve)
        outer_cv_results["best_train"].append(
            cv_clf.cv_results_["mean_train_score"][cv_clf.best_index_]
        )
        outer_cv_results["best_val"].append(cv_clf.cv_results_["mean_test_score"][cv_clf.best_index_])
        if name_optim_file:
            dims = []
            for param in param_grid.keys():
                if len(param.split('__')) > 1:
                    dims.append(param.split('__')[1])
            try:
                _ = plot_objective(cv_clf.optimizer_results_[0], dimensions=dims)
            except:
                _ = plot_objective(cv_clf.optimizer_results_[0])
            optim_filepath = "{}optimizer_{}_{}.png".format(filename_base, name_optim_file, i)
            plt.savefig(optim_filepath)

        if not nested:
            print("\t\t\tBest train: {}".format(outer_cv_results["best_train"]))
            print("\t\t\tBest val: {}".format(outer_cv_results["best_val"]))

            outer_cv_results["best_estimator"] = cv_clf.best_estimator_
            outer_cv_results["best_params"] = cv_clf.cv_results_["params"][cv_clf.best_index_]
            outer_cv_results["columns"] = X_train.columns

            break

        # Performance metrics on test set
        y_pred = cv_clf.predict(X_test)
        try:  # SVM
            y_pred_decision = cv_clf.decision_function(X_test)
        except AttributeError:  # RF and XGB
            y_pred_decision = cv_clf.predict_proba(X_test)[:, 1]

        inner_cv_performances = {
            "auc": metrics.roc_auc_score(y_test, y_pred_decision),
            "precision": metrics.precision_score(y_test, y_pred),
            "recall": metrics.recall_score(y_test, y_pred),
            "roc_curve": metrics.roc_curve(y_test, y_pred_decision),
        }
        for name, result in inner_cv_performances.items():
            outer_cv_results[name].append(result)

        # Print results for loop
        print("\t\t\tOuter loop {} done in {} seconds".format(i + 1, int(duration)))
        print("\t\t\t\tAUC: {}".format(outer_cv_results["auc"]))
        print("\t\t\t\tPrecision: {}".format(outer_cv_results["precision"]))
        print("\t\t\t\tRecall: {}".format(outer_cv_results["recall"]))
        print("\t\t\t\tBest train: {}".format(outer_cv_results["best_train"]))
        print("\t\t\t\tBest val: {}".format(outer_cv_results["best_val"]))

        # Dataframe for CI on AUC
        try:
            subject_index = [i for i, _ in index_test.tolist()]
        except (ValueError, TypeError) as e:
            subject_index = index_test.tolist()

        prediction_df = pd.DataFrame({"subject_id": subject_index,
                                      "y_pred": y_pred_decision,
                                      "y_true": y_test,
                                      "cv_fold": i})
        outer_cv_results["prediction_df"].append(prediction_df)

    # get 95% CI on AUC scores
    if nested:
        outer_cv_results["predictions_cv"] = pd.concat(outer_cv_results["prediction_df"], ignore_index=True)

    return outer_cv_results


def start_hyperparameter_tuning(data_params, params_clf, df, output_filepath="cv_results.csv"):
    """ Start hyper parameter tuning process with configurations from `param_grid` using `model`.
    Each hyperparameter configuration undergoes a leave-one-participant out cross validation loop.

    Args:
    ---
    - `param_grid`: Dict[str,List[Any]]
      The param grid to perform grid search.
    - `df`: DataFrame
      The dataset to perform cross validation on.
    - `model`: ClassifierMixin
      The sklearn classification model to train.
    - `output_filepath`: str, Optional
      The path to save output file. (Default is cv_results.csv)
    """
    all_params = list(product(*[v for _, v in params_clf.items()]))

    fieldnames = list(params_clf.keys())  + \
                 ['win_size', 'win_stride', 'features'] + \
                 ["accuracy", "recall_score", "precision_score", "f1_score", "tn", "fp", "fn", "tp", 'auc',
                  "split_val", "split_train"]

    with open(output_filepath, 'a+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()
        csvfile.close()

    cv_splits = list(leave_one_participant_out_cv(df))

    outter_pbar = tqdm(all_params, desc="Configurations")
    inner_pbar = tqdm(cv_splits, desc=f"CV", leave=False)


    for clf_params in params_clf:
        split_results = []
        for train_participants, validation_participant in cv_splits:
            train = df[df['participant'].isin(train_participants)]
            train = train.drop(['timestamp', 'participant'], axis=1)
            X_train, y_train = preprocess(train.drop(['stress'], axis=1), train['stress'], **data_params)

            validation = df[df['participant'].isin([validation_participant])]
            validation = validation.drop(['timestamp', 'participant'], axis=1)
            X_val, y_val = preprocess( validation.drop(['stress'], axis=1), validation['stress'], **data_params)

            clf = clf_params['estimator'](**clf_params)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_val)

            acc, precision, recall, f1, tn, fp, fn, tp = evaluate(y_val, y_pred)

            evaluations = {
                "accuracy": acc,
                "recall_score": recall,
                "precision_score": precision,
                "f1_score": f1,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }

            sets = {
                "split_val": validation_participant,
                "split_train": ",".join(train_participants)
            }

            split_results.append(dict(params, **evaluations, **sets))

            inner_pbar.update(1)

        inner_pbar.reset()

        with open(output_filepath, 'a+') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(split_results)
            csvfile.flush()
            csvfile.close()

        outter_pbar.update(1)


def evaluate(y_true, y_pred):
    """ Calculates accuracy, precision, recall, f1, tn, fp, fn, tp from `y_true`, `y_pred`.

    Args:
    ---
    - `y_true`: List[Any]
      List of target value.
    - `y_pred`: List[Any]
      List of predicted value that are the output of classifier.

    Returns:
    ---
    Tuple[float, float, float, float, int, int, int, int, int]
      Evaluations output in the form of (accuracy, precision, recall, f1, tn, fp, fn, tp)
    """
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return acc, precision, recall, f1, tn, fp, fn, tp


def read_test_result(filepath, param_grid):
    """ Read the test result file.

    Args:
    ---
    - `filepath`: str
      The path to test file.
    - `param_grid`: Dict[str, Any]
      The param configurations to try.

    Returns:
    ---
    DataFrame
      The test result DataFrame.
    """
    result = pd.read_csv(filepath).drop(['tp', 'fp', 'tn', 'fn'], axis=1)\
        .groupby(list(param_grid.keys()), dropna=False).agg(['mean', 'std']).reset_index()
    result.columns = ["_".join(c) if c[-1] != "" else c[0] for c in result.columns.to_flat_index()]
    return result


def save_cv_results(shelve_path, future=False, test=True):
    predictions_cv = []
    with shelve.open(shelve_path) as cv_results:
        for name, cv in cv_results.items():
            print(f"\t\t\tName: {name};\tShelve_path: {shelve_path}")
            cv["predictions_cv"]["clf"] = name
            predictions_cv.append(cv["predictions_cv"])

    predictions_cv = pd.concat(predictions_cv, ignore_index=True)
    predictions_cv["outcome"] = 'stress'

    pred_filename_elements = ["predictions_stress"]
    if test:
        pred_filename_elements.append("test")
    pred_filename = "_".join(pred_filename_elements)
    if future:
        predictions_cv.to_feather("Results_Future/Models/{}.feather".format(pred_filename))
    else:
        predictions_cv.to_feather("Results/Models/{}.feather".format(pred_filename))
    print("\t\t\tPredictions saved to file: {}".format(pred_filename))
