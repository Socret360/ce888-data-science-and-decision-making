# Project Description
Stress Prediction with 

# Table of Content
1. [Requirements](#1-requirements)
2. [Project Structure](#2-project-structure)
3. [How to run](#3-how-to-run)
    
    3.1. [Re-run experiment](#31-re-run-experiments)
    
    3.2. [Inference](#32-inference)

# 1. Requirements
This project is developed using Python version 3.8.5. It relies on the following third party libraries (also listed in `requirements.txt`):

```shell
tqdm===4.65.0          # graphical progress bars.
pandas===1.5.3         # read/write/manipulate tabular data.
jupyter===1.0.0        # combining python codes with markdown explanations.
seaborn===0.12.2       # creating nice looking graphs
matplotlib===3.7.1     # used by seaborn as backend visualisation engine.
scikit-learn===1.2.2   # developing machine learning models.
```

# 2. Project Structure
```shell
project
│───assets            # Images of graphs and figures.
|───data              # Raw and processed data.
|───output            # Output of experiments.
|   experiment.ipynb  # Jupyter notebook of main experiment.
|   README.md         # Project Documenttation.
|   requirements.txt  # Requirements file.
|   utils.py          # Utils functions used in `experiments.ipynb`.
```

# 3. How to run
It is recommended to create a virtual environment using conda, virtualenv, etc before going through the following instructions.

## 3.1. Re-run experiments
This section outlines instructions for re-runing the experiment.

1. Make sure you are in project directory.
```shell
pwd
# ce888-data-science-and-decision-making/project
```

2. Install the needed third parties packages:
```shell
pip install -r requirements.txt
```

3. Download [Raw_data.zip](https://drive.google.com/file/d/1LE89wFp0jufWcDYYh29l1y1zldOdDbk6/view?usp=share_link) and extract it to `project/data/`. This is a modified version of the original data introduced in [italha-d/Stress-Predict-Dataset](https://github.com/italha-d/Stress-Predict-Dataset). In particular, it fixes the error in [task_S27_tags.csv](https://github.com/italha-d/Stress-Predict-Dataset/blob/main/Raw_data/S27/tags_S27.csv) of participant S27.

4. Start jupyter notebook server.
```shell
jupyter notebook
```

5. Open up [experiment.ipynb](./experiment.ipynb).
6. In the `SETUP` section of the notebook, edit the `PROJECT_DIR` variable if you place [experiment.ipynb](./experiment.ipynb) in a different directory from the `ce888-data-science-and-decision-making/project`.

![jupyter-example-setup](./assets/jupyter-example-setup.png)

## 3.2. Inference
This project provides [inference.py](./inference.py) for running inferences using the trained model from [experiment.ipynb](./experiment.ipynb).

1. 

# 4. References
