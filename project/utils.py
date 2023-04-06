import os
import numpy as np
import pandas as pd
import seaborn as sns
from functools import reduce
import matplotlib.pyplot as plt
from itertools import islice, chain, repeat


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
    combined, start_time, end_time = combine_participant_measurements(
        participant, sampling_rate)
    task_tags = read_participant_task_tags(os.path.join(
        participant, f"tags_{os.path.basename(os.path.normpath(participant))}.csv"))
    tagged = determine_timestamp_stress_status_of_participant(
        combined, task_tags)
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
    - `leading`: int, Optional. Default to 240
      The number of seconds before the first stress period.
    - `trailing`: int, Optional. Default to 240
      The number of seconds after the last stress period.

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


def moving_window(df, window_size=64, step=1):
    """ Sliding window generator for moving across `df`.

    Args:
    ---
    - `df`: DataFrame
        The DataFrame to slide across.
    - `window_size`: int, Optional. Defaults to 64 samples
        The number of samples per window.
    - `step`: int, Optional. Defaults to 1 sample.
        The step size to move forward.

    Returns:
    ---
    Generator[DataFrame, Series]
        A generator that yields a window and its label.
    """
    for idx in range(0, len(df)-window_size, step):
        yield df.iloc[idx:idx+window_size], df.iloc[idx+window_size]


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
