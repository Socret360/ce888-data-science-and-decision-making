"""Run inference from user input.

This script takes user input in the form of path to a csv file from command line and run inferences on the data points in the csv file.
The csv file should have the following columns:
- accel_x
- accel_y
- accel_z
- skin_temp
- heart_rate
- blood_volume_pulse
- eda
- inter_beat_interval


Author: Socretquuliqaa <lee.socret@gmail.com>
Created: 09/04/2023
"""

import os
import json
import pickle
import argparse
import pandas as pd

import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run inference from user input.')
    parser.add_argument('input', type=str, help='path to input csv file.')
    parser.add_argument('model_path', type=str, help='path to saved model folder that contains both the pickle and json files.')
    parser.add_argument('--output_dir', type=str,
                        help='path to output directory.', default=None)
    args = parser.parse_args()

    with open(os.path.join(args.model_path, "model.json"), 'r') as config_file:
        config = json.load(config_file)

    input_data = pd.read_csv(args.input)
    input_data = input_data[['accel_x', 'accel_y', 'accel_z', 'skin_temp', 'heart_rate', 'blood_volume_pulse', 'eda', 'inter_beat_interval']]

    X, _ = utils.preprocess(
        input_data,
        None,
        window_size=config['window_size'],
        stride=config['stride'],
    )

    clf = pickle.load(open(os.path.join(args.model_path, "model.pkl"), 'rb'))
    y_pred = clf.predict(X)

    if args.output_dir is not None:
        # save the prediction to a csv file.
        pd.DataFrame(
            columns=['Stress'],
            data=y_pred
        ).to_csv(os.path.join(args.output_dir, "pred.csv"), index=False)
    else:
        print(y_pred)
