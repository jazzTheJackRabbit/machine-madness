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
dataset = pd.read_csv(ROOT_DIR+"data/structured/training_data_match_statistics.csv")
dataset = dataset.drop(dataset.columns[0],axis=1)

def getPredictionsFromClassifier(classifier,dataset):
    # Training
    training_data = dataset[dataset.season != 2014]
    testing_data = dataset[dataset.season == 2014]

    #Preprocess/Filter Data here

    training_X = training_data[training_data.columns[1:len(training_data.columns)-1]].fillna(0)
    training_Y = training_data[training_data.columns[-1]]

    testing_X = testing_data[testing_data.columns[1:len(testing_data.columns)-1]].fillna(0)
    testing_Y = testing_data[testing_data.columns[-1]]
    classifier.fit(training_X,training_Y)
    predicted_Y = classifier.predict(testing_X)

    print(classification_report(predicted_Y.round().astype(int),testing_Y))
    print(accuracy_score(predicted_Y.round().astype(int),testing_Y))

    return predicted_Y

def classNameForClassifier(classifier):
    class_name = str(classifier.__class__)
    class_name = class_name[class_name.rfind(".")+1:class_name.rfind("'>")]
    return class_name

def addPredictionAndClassifierToList(predictions,classifier,predicted_Y):
    class_name = classNameForClassifier(classifier)
    prediction = pd.DataFrame(predicted_Y,columns=[class_name])
    predictions = pd.concat([predictions,prediction],axis=1)
    import pdb; pdb.set_trace()  # breakpoint 9d24d050 //    
    return predictions

dataset = pd.read_csv(ROOT_DIR+"data/structured/training_data_match_statistics.csv")
dataset = dataset.drop(dataset.columns[0],axis=1)

classifiers = [RandomForestRegressor(),LogisticRegression()]
predictions = pd.DataFrame()

for classifier in classifiers:
	print(classifier.__class__)
	predicted_Y = getPredictionsFromClassifier(classifier,dataset)
	predictions = addPredictionAndClassifierToList(predictions,classifier,predicted_Y)
