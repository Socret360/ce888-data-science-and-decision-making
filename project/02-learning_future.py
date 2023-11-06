import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import seaborn as sns
import pandas as pd
import shelve
import numpy as np
import os
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt.space import Real, Categorical, Integer
from joblib import effective_n_jobs

import utils  # Our library


PROJECT_DIR = './'  # path to the project directory
DATA_DIR = os.path.join(PROJECT_DIR, 'data/')  # path to the data directory
SHELVE_PATH = 'Results_Future/Models/'  # To store results

EVALUATION_COLUMNS = ['accuracy', 'recall_score', 'precision_score', 'f1_score', 'tn', 'fp', 'fn', 'tp', 'auc']

SRATE = 32  # Hz; sampling rate at which the data were saved
NEW_SRATE = 16


def main():
    print('Number of jobs:', effective_n_jobs(-1))
    # Load data
    subsample = SRATE // NEW_SRATE
    train_set = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))[::subsample]  # downsample to NEW_SRATE
    test_set = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))[::subsample]  # downsample to NEW_SRATE

    params_data = {
        'window_size': [int(NEW_SRATE * i) for i in [30, 60, 120, 180, 300]],
        'features': utils.get_all_combinations(['ACC', 'IBI', 'TEMP', 'EDA', 'HR']),
        'future': [int(NEW_SRATE * j) for j in [10, 15, 30, 60, 90, 120]]  # predict stress in the future
    }

    roc_auc = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

    # Hyperparameter tuning and cross-validation
    params_clfs_bayes = [
        ('XGBoost', {'clf': [XGBClassifier()], 'clf__max_depth': Integer(1, 20, prior='uniform'),
                     'clf__n_estimators': Integer(20, 400, prior='uniform'),
                     'clf__reg_alpha': Real(0.0001, 10, prior='log-uniform'),
                     'clf__reg_lambda': Real(0.01, 100, prior='log-uniform'),
                     'clf__learning_rate': Real(5e-4, 0.5, prior='log-uniform'),
                     'clf__subsample': Real(0.3, 1, prior='uniform')}),
        ('randomForest', {'clf': [RandomForestClassifier()], 'clf__n_estimators': Integer(10, 100, prior='uniform')}),
        ('decisionTree', {'clf': [DecisionTreeClassifier()], 'clf__criterion': Categorical(['entropy']),
         'clf__max_depth': Integer(1, 10, prior='uniform')}),
        ('SVC_rbf', {'clf': [SVC()], 'clf__C': Real(1e-5, 1000, prior='uniform'), 'clf__kernel': ['rbf'],
                     'clf__gamma': Real(0.0001, 0.9, prior='log-uniform')}),
        ('SVC_linear', {'clf': [SVC()], 'clf__C': Real(1e-5, 1000, prior='log-uniform'),
                        'clf__kernel': Categorical(['linear'])}),
        ('LDA', {'clf': [LDA()], 'clf__shrinkage': Real(0, 1, prior='uniform'),
                 'clf__solver': Categorical(['lsqr', 'eigen'])}),
        #('LogisticRegression', {'clf': [LogisticRegression()], 'clf__C': Real(1e-5, 10, prior='log-uniform'),
        #                        'clf__penalty': Categorical([None, 'l1', 'l2']),
        #                        'clf__solver': Categorical(['saga'])})
        #('LGBMClassifier', {'clf': [LGBMClassifier()], 'clf__learning_rate': Real(0.0005, 0.3, prior='log-uniform'),
        #                    'clf__subsample': Real(0.3, 1, prior='uniform'),
        #                    'clf__n_estimators': Integer(500, 1000, prior='uniform'),
        #                    'clf__max_depth': Integer(3, 15, prior='uniform'),
        #                    'clf__num_leaves': Integer(20, 100, prior='uniform'),
        #                    'clf__reg_alpha': Real(0.0001, 10, prior='log-uniform'),
        #                    'clf__reg_lambda': Real(0.01, 100, prior='log-uniform')})
    ]
    inner_cv_n_split = 7
    outer_cv_n_split = 4
    for features in params_data['features']:
        for window_size in params_data['window_size']:  # samples
            for stride_percentage in [.25, .5, .75, 1]:
                stride = int(stride_percentage * window_size)  # samples
                if stride/NEW_SRATE < 1:
                    continue
                for future in params_data['future']:
                    if future > window_size:
                        continue
                    name = 'winSize_{}_stride_{}_future_{}_features_'.format(window_size / NEW_SRATE,  # seconds
                                                                             stride / NEW_SRATE, future / NEW_SRATE)
                    if len(features) > 1:
                        name += ','.join(feat for feat in features)
                    else:
                        name += features[0]
                    shelve_path = '{}cv_dict_{}'.format(SHELVE_PATH, name)
                    print('shelve path:', shelve_path)
                    with shelve.open(shelve_path) as cv_results:
                        print('Starting...', name)
                        input_for_cv = utils.prepare_data(train_set, inner_cv_n_split=inner_cv_n_split,
                                                          outer_cv_n_split=outer_cv_n_split,
                                                          params={'window_size': window_size, 'stride': stride,
                                                                  'features': features, 'future': future},
                                                          test=True, df_test=test_set)
                        input_for_cv['shelve_path'] = shelve_path
                        print('\tData ready')
                        for name_clf, clf_params in params_clfs_bayes:
                            full_name = '{}_{}'.format(name_clf, name)
                            print('\tStarting...', name_clf)
                            pipeline = Pipeline([('std', StandardScaler()), ('clf', clf_params['clf'])])
                            # Inner CV
                            print('\t\tRunning loop...')
                            cv_results[full_name] = utils.fit_bayes_cv(input_for_cv['X'], input_for_cv['y'],
                                                                       pipeline=pipeline, groups=input_for_cv['groups'],
                                                                       inner_cv=input_for_cv['inner_cv'],
                                                                       outer_cv=input_for_cv['outer_cv'],
                                                                       n_bayes_iter=100, param_grid=clf_params,
                                                                       scoring=roc_auc, n_jobs=effective_n_jobs(-1),
                                                                       name_optim_file=full_name,
                                                                       filename_base='Results_Future/Models/Optim/')
                        # Predictions
                        utils.save_cv_results(input_for_cv['shelve_path'], future=True, test=True)


if __name__ == "__main__":
    main()



