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

def getPredictionsFromClassifier(classifier,training_X,training_Y,testing_X,testing_Y):
    # Training    
    classifier.fit(training_X,np.ravel(training_Y))
    predicted_Y = classifier.predict(testing_X)

    if(hasattr(classifier,"predict_proba")):
        predicted_probabilities_Y = classifier.predict_proba(testing_X)[:,0]
    else:
        predicted_probabilities_Y = predicted_Y

    print(classification_report(predicted_Y.round().astype(int),testing_Y))
    print(accuracy_score(predicted_Y.round().astype(int),testing_Y))

    return predicted_Y,predicted_probabilities_Y

def classNameForClassifier(classifier):
    class_name = str(classifier.__class__)
    class_name = class_name[class_name.rfind(".")+1:class_name.rfind("'>")]
    return class_name

def addPredictionAndClassifierToList(prediction_probabilities_for_winning,classifier,predicted_probabilities_Y):
    class_name = classNameForClassifier(classifier)
    prediction = pd.DataFrame(predicted_probabilities_Y,columns=[class_name])
    prediction_probabilities_for_winning = pd.concat([prediction_probabilities_for_winning,prediction],axis=1)
    return prediction_probabilities_for_winning

# Dataset Partitioning
dataset = pd.read_csv(ROOT_DIR+"data/structured/training_data_match_statistics.csv")
dataset = dataset.drop(dataset.columns[0],axis=1)

target_season = 2014
training_data = dataset[(dataset.season != target_season) & (dataset.season > target_season-4)]
testing_data = dataset[dataset.season == target_season]

#Preprocess/Filter Data here
startingFeatureIndex = 3

training_X = training_data[training_data.columns[startingFeatureIndex:len(training_data.columns)-1]].fillna(0)
training_Y = training_data[training_data.columns[-1]]

testing_X = testing_data[testing_data.columns[startingFeatureIndex:len(testing_data.columns)-1]].fillna(0)
testing_Y = testing_data[testing_data.columns[-1]]

print("***************************")
print("EVALUATION ON TRAINING DATA")
print("***************************")

testing_X = training_X
testing_Y = training_Y

classifiers = [RandomForestClassifier(),LogisticRegression()]
prediction_probabilities_for_winning = pd.DataFrame()

for classifier in classifiers:
    predicted_Y, predicted_probabilities_Y = getPredictionsFromClassifier(classifier,training_X,training_Y,testing_X,testing_Y)
    prediction_probabilities_for_winning = addPredictionAndClassifierToList(prediction_probabilities_for_winning,classifier,predicted_probabilities_Y)   
    
training_Y = pd.DataFrame(training_Y.as_matrix(),columns=["winningTeam"])

ensemble_training_X = prediction_probabilities_for_winning
ensemble_training_Y = training_Y

print("***************************")
print("EVALUATION ON TESTING DATA")
print("***************************")

testing_X = testing_data[testing_data.columns[startingFeatureIndex:len(testing_data.columns)-1]].fillna(0)
testing_Y = testing_data[testing_data.columns[-1]]

prediction_probabilities_for_winning = pd.DataFrame()

for classifier in classifiers:
    predicted_Y, predicted_probabilities_Y = getPredictionsFromClassifier(classifier,training_X,training_Y,testing_X,testing_Y)
    prediction_probabilities_for_winning = addPredictionAndClassifierToList(prediction_probabilities_for_winning,classifier,predicted_probabilities_Y)   

# 
# ENSEMBLE CLASSIFIER
# 

ensemble_testing_X = prediction_probabilities_for_winning
ensemble_testing_Y = testing_Y    

print("***************************")
print("ENSEMBLE")
print("***************************")

ensemble_classifier = LogisticRegression()
ensemble_classifier.fit(ensemble_training_X,np.ravel(ensemble_training_Y))

y_pred = ensemble_classifier.predict(ensemble_testing_X)
print(classification_report(y_pred,ensemble_testing_Y))
print(accuracy_score(y_pred,ensemble_testing_Y))

y_pred_prob =  pd.DataFrame(ensemble_classifier.predict_proba(ensemble_testing_X)[:,0], columns=["pred"])
df = pd.DataFrame(columns=["id"])
df['id'] = str(target_season)+"_"+testing_data.team1.map(str)+"_"+testing_data.team2.map(str)

# Remove weird index
df = df.reset_index().drop("index",axis=1)

output = pd.concat([df['id'],y_pred_prob],axis=1)
output.to_csv(ROOT_DIR+"data/structured/prediction_probabilities_for_matchups.csv")
