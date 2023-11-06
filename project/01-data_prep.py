import os

import pandas as pd
from glob import glob
from tqdm.auto import tqdm

# util functions
import utils

# display floating points in pandas with 5 digits after decimal points
pd.options.display.float_format = '{:.5f}'.format

PROJECT_DIR = "./"  # path to the project directory
DATA_DIR = os.path.join(PROJECT_DIR, "data/")  # path to the data directory
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output/")  # path to store outputs produced during training and evaluation.
DATASET_PATH = os.path.join(DATA_DIR, "Raw_data/")  # path the raw data directory


def main(crop=False):
    srate = 32
    # create the data and output dir if it does not exist yet.
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # For each participant, 6 measurements are stored in 6 different files:
    # `ACC.csv`, `BVP.csv`, `EDA.csv`, `HR.csv`, `IBI.csv`, `TEMP.csv`.
    # Each of these measurements has different sampling rate but the sampling rate of a measurement
    # is the same across all participants.
    # In addition, `tag_SX.csv` is also provided for determining the `start_time` and `end_time` of events
    # performed by the participant.
    # The first 3 events are the stress inducing events.

    try:
        data = pd.read_csv(os.path.join(DATA_DIR, "clean_data.csv"), na_values='NA')
    except:

        # Combine all measurements from all participants together
        dataframes = []

        for participant in tqdm(sorted(list(glob(os.path.join(DATASET_PATH, "S*"))))):
            df, _, _ = utils.read_participant_measurements(participant, sampling_rate=srate)
            df['participant'] = participant.split(os.sep)[-1]
            dataframes.append(df)

        result = pd.concat(dataframes)

        # reorder columns so that the first two columns are: participant and timestamp
        leading_columns = ['participant', 'timestamp']
        reordered_columns = leading_columns + [col for col in result.columns.tolist() if col not in leading_columns]

        # My final tidy data consists of 11 signals.
        # participant: str, The id of the participant.
        # timestamp: float, seconds The UNIX format timestamp of the entry.
        # accel_x: float, gravitational force g The x value of the accelerometer.
        # accel_y: float, gravitational force g The y value of the accelerometer.
        # accel_z: float, gravitational force g The z value of the accelerometer.
        # skin_temp: float, Celsius The skin temperature.
        # heart_rate: float The average heart rate calculated from raw BVP value.
        # blood_volume_pulse: float, nanowatts The raw BVP data.
        # eda: float, The skin conductance.
        # inter_beat_interval: float, seconds The IBI values.
        # stress: bool, indicates whether the participant is performing stress-inducing tasks.

        data = data.fillna(method='ffill')  # for the last timestamps with Nan, fill in with last known.
        data = data.fillna(method='bfill')  # for the first timestamps with NaN, fill in with first known.

        data.to_csv(os.path.join(DATA_DIR, "clean_data.csv"), index=False)

    participants = data['participant'].unique().tolist()

    if crop:
        # There's class imbalance.
        # To deal with this, we trim the leading and trailing non-stress period.
        # More specifically, for each participant, I kept the maximum leading and trailing non-stress period to be 4 mins.
        leading = 4 * 60
        results = []
        for participant in data['participant'].unique().tolist():
            clipped = utils.clip_participant_experiment_data(data[data['participant'] == participant],
                                                             leading=leading, trailing=leading)
            results.append(clipped)
    data = pd.concat(results)

    # Split participants into training and test sets
    # The last 7 participants are kept for testing
    train_participants, test_participants = participants[:-7], participants[-7:]
    train_set = data[data['participant'].isin(train_participants)]
    test_set = data[data['participant'].isin(test_participants)]

    train_set.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    test_set.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)


if __name__ == "__main__":
    main(crop=True)
