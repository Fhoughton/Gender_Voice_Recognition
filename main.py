import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

features = pd.read_csv('voice.csv')

#convert genders to 0 and 1
for i in range(len(features["label"])):
    if features["label"][i] == "male":
        features.loc[:,("label")][i] = 1
    else:
        features.loc[:,("label")][i] = 0

#Store values we want to predict
labels = np.array(features["label"])

#Save feature names for later use
feature_list = list(features.columns)

#Convert to numpy array
features = np.array(features)

#Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, 
                                                                            random_state = 19)
#Establish a baseline prediction to beat
baseline_preds = np.full((792, 1), 0, dtype=int) #assumes all female

baseline_errors = abs(baseline_preds - test_labels)

print('Average baseline error: ', round(np.mean(baseline_errors), 2)) #Average baseline error:  0.49

#Create the model
rfr = RandomForestRegressor(n_estimators = 1250, random_state = 97)

#Train the model
rfr.fit(train_features, train_labels);

#Make a prediction on the test data
predictions = rfr.predict(test_features)

errors = abs(predictions - test_labels)

#Calculate accuracy using mean absolute percentage error
print('Average model error: ', round(np.mean(errors), 2)) #Average baseline error:  0.49
