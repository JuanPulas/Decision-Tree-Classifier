import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

# Step 1 - Get the file
dirname = os.path.dirname(__file__)
heart_file = os.path.join(dirname, "heart.csv")
heart = pd.read_csv(heart_file)
# print(heart)
# print(heart.head())

# Step 2 - A. Transform features | B. Determine x and y axis
# For experimenting/learning purposes, do it manually. Otherwise, use sklearn OneHotEncode()
# encoder = OneHotEncoder(handle_unknown='ignore')
# encoder.fit(X_data) # this line must be placed after the X_data declaration
heart["HeartDisease"] = heart["HeartDisease"].replace({'No': 0, 'Yes': 1})
heart["Smoking"] = heart["Smoking"].replace({'No': 0, 'Yes': 1})
heart["AlcoholDrinking"] = heart["AlcoholDrinking"].replace({'No': 0, 'Yes': 1})
heart["Stroke"] = heart["Stroke"].replace({'No': 0, 'Yes': 1})
heart["Sex"] = heart["Sex"].replace({'Male': 0, 'Female': 1})
heart['AgeCategory'] = heart['AgeCategory'].replace(['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74','75-79','80 or older'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
# heart['Race'] = heart['Race'].replace(['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Other', 'Hispanic'], [1, 2, 3, 4, 5, 6])
heart["Diabetic"] = heart["Diabetic"].replace(['No', 'Yes', 'No, borderline diabetes', 'Yes (during pregnancy)'], [1,2,3,4])
# X_data = heart[['BMI','Smoking','AlcoholDrinking','Stroke','PhysicalHealth','Sex','AgeCategory', 'Race', 'Diabetic']]
X_data = heart[['BMI','Smoking','AlcoholDrinking','Stroke', 'Sex','AgeCategory', 'Diabetic']]
y_data = heart['HeartDisease']


# Step 3 - split training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=36)

# Step 4 - import & use the decision tree model from scikit learn to fit/train the model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(min_samples_leaf=500)
model.fit(X_train, y_train)


# Step 5 - Genrate Graph
heart["HeartDisease"] = heart["HeartDisease"].replace({0 :'No', 1 : 'Yes'})
from sklearn.tree import export_graphviz
export_graphviz(model, 'tree.dot', feature_names = ['BMI','Smoking','AlcoholDrinking','Stroke','Sex','AgeCategory', 'Diabetic'], class_names=heart['HeartDisease'].unique())

score = model.score(X_train, y_train)
print("Score: ", score)

pred = model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))
