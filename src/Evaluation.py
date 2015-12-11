from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm, grid_search
import pandas as pd
import numpy as np

# 
# Creates MATCH statistics for all tournament matches in a season (for all seasons)
# 

ROOT_DIR = "../"

def computeLogLoss(submission_Y,testing_Y):
    y = testing_Y
    log_y_hat = np.log(submission_Y)

    _1_y = 1 - testing_Y
    _log_1_y_hat = np.log(1.0 - submission_Y)

    logLoss = (-1.0/y.shape[0]) * (np.dot(y.transpose(),log_y_hat) + np.dot(_1_y.transpose(),_log_1_y_hat))
    return logLoss

# Dataset Partitioning
dataset = pd.read_csv(ROOT_DIR+"data/structured/training_data_match_statistics.csv")
submission = pd.read_csv(ROOT_DIR+"data/structured/prediction_probabilities_for_matchups.csv")

target_season = 2014
dataset = dataset.drop(dataset.columns[0],axis=1)
testing_data = dataset[dataset.season == target_season]

#Preprocess/Filter Data here
startingFeatureIndex = 3

testing_Y = testing_data[testing_data.columns[-1]].reset_index().drop("index",axis=1).as_matrix()
submission_Y = submission[submission.columns[-1]].reset_index().drop("index",axis=1).as_matrix()

logLoss = computeLogLoss(submission_Y, testing_Y)

print(logLoss)