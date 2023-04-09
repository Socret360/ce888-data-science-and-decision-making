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

import argparse
import pandas as pd

import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run inference from user input.')
    parser.add_argument('input', type=str, help='path to input csv file.')
    parser.add_argument('model_path', type=str, help='path to saved model.')
    parser.add_argument('--output_dir', type=str,
                        help='path to output directory.', default="output")
    args = parser.parse_args()
