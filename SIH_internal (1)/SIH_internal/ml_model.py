import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imblearn
from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


from sklearn.metrics import accuracy_score
import pickle

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("structured_water_footprint_dataset.csv")
#print(data.head())
occupation_type_label_encoder = LabelEncoder()
data["Occupation"] = occupation_type_label_encoder.fit_transform(data["Occupation"])

Lifestyle_type_label_encoder = LabelEncoder()
data["Lifestyle"] = Lifestyle_type_label_encoder.fit_transform(data["Lifestyle"])

garden_type_label_encoder = LabelEncoder()
data["Presence_of_garden"] = garden_type_label_encoder.fit_transform(data["Presence_of_garden"])

dish_type_label_encoder = LabelEncoder()
data["Dishwasher_usage"] = dish_type_label_encoder.fit_transform(data["Dishwasher_usage"])

wash_type_label_encoder = LabelEncoder()
data["Washing_machine_usage"] = wash_type_label_encoder.fit_transform(data["Washing_machine_usage"])

swim_type_label_encoder = LabelEncoder()
data["Swimming_pool"] = swim_type_label_encoder.fit_transform(data["Swimming_pool"])

water_type_label_encoder = LabelEncoder()
data["Water_storage"] = water_type_label_encoder.fit_transform(data["Water_storage"])




#Presence_of_garden
#Dishwasher_usage
#Washing_machine_usage
#Swimming_pool
#Water_storage



occupation_dict = {}
for i in range(len(data["Occupation"].unique())):
    occupation_dict[i] = occupation_type_label_encoder.inverse_transform([i])[0]
#print(croptype_dict)
print(occupation_dict)
life_dict = {}
for i in range(len(data["Lifestyle"].unique())):
    life_dict[i] = Lifestyle_type_label_encoder.inverse_transform([i])[0]
print(life_dict)
garden_dict = {}
for i in range(len(data["Presence_of_garden"].unique())):
    garden_dict[i] = garden_type_label_encoder.inverse_transform([i])[0]
print(garden_dict)
Dish_dict = {}
for i in range(len(data["Dishwasher_usage"].unique())):
    Dish_dict[i] = dish_type_label_encoder.inverse_transform([i])[0]
print(Dish_dict)
wash_dict = {}
for i in range(len(data["Washing_machine_usage"].unique())):
    wash_dict[i] = wash_type_label_encoder.inverse_transform([i])[0]
print(wash_dict)
swim_dict = {}
for i in range(len(data["Swimming_pool"].unique())):
    swim_dict[i] = swim_type_label_encoder.inverse_transform([i])[0]
print(swim_dict)
water_dict = {}
for i in range(len(data["Water_storage"].unique())):
    water_dict[i] = water_type_label_encoder.inverse_transform([i])[0]
print(water_dict)





# Water_footprint_label_encoder = LabelEncoder()
# data["Water_footprint"] = Water_footprint_label_encoder.fit_transform(data["Water_footprint"])


#print(fertname_dict)

#print(data.head())

X = data[data.columns[:-1]]
y = data[data.columns[-1]]

#Upscaling the data

counter = Counter(y)
# #print(counter)
# #print(Counter(y))

upsample = SMOTE()

X, y = upsample.fit_resample(X, y)
# counter = Counter(y)
#print(counter)
print(f"Total Data after Upsampling: {len(X)}")#154
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size = 0.2, random_state = 0)
#print(f"Train Data: {X_train.shape}, {y_train.shape}")
#print(f"Train Data: {X_test.shape}, {y_test.shape}")

#=================Random-ForestClassifier=========================================
rf_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state = 18))
rf_pipeline.fit(X_train, y_train)
predictions = rf_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy on Test Data: {accuracy*100}%")
predict = rf_pipeline.predict(X.values)
accurac = accuracy_score(y, predict)
print(f"Accuracy on Total Data: {accurac*100}%")
pickle.dump(rf_pipeline, open("rf_pipeline.pkl", "wb"))

'''X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size = 0.2, random_state = 42)
#print(f"Train Data: {X_train.shape}, {y_train.shape}")
#print(f"Train Data: {X_test.shape}, {y_test.shape}")

knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = 1))
knn_pipeline.fit(X_train, y_train)

predictions = knn_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy on Test Data: {accuracy*100}%")

predict = knn_pipeline.predict(X.values)
accuracy = accuracy_score(y, predict)
print(f"Accuracy on Whole Data: {accuracy*100}%")


pickle.dump(knn_pipeline, open("knn_pipeline.pkl", "wb"))'''

