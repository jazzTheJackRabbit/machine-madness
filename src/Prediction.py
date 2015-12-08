from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model  import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# 
# Creates MATCH statistics for all tournament matches in a season (for all seasons)
# 

ROOT_DIR = "../"
dataset = pd.read_csv(ROOT_DIR+"data/structured/training_data_match_statistics.csv")
dataset = dataset.drop(dataset.columns[0],axis=1)

classifier = RandomForestClassifier(n_estimators=200)
# classifier = LogisticRegression(C=1e6)
# classifier = LinearRegression()


training_data = dataset[dataset.season != 2014]
testing_data = dataset[dataset.season == 2014]

#Preprocess/Filter Data here

training_X = training_data[training_data.columns[1:len(training_data.columns)-1]].fillna(0)
training_Y = training_data[training_data.columns[-1]]

testing_X = testing_data[testing_data.columns[1:len(testing_data.columns)-1]].fillna(0)
testing_Y = testing_data[testing_data.columns[-1]]

# Training
classifier.fit(training_X,training_Y)
predicted_Y = classifier.predict(testing_X)

print(classification_report(predicted_Y.round().astype(int),testing_Y))
print(accuracy_score(predicted_Y.round().astype(int),testing_Y))