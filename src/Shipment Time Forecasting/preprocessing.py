# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 14:41:34 2023

@author: Nikhil
"""

# importing the libraries
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# creating the class
class Preprocessing:

    # constructor
    def __init__(self, dataset):
        self.dataset = dataset
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    # label encoding
    def label_encode(self, column_name):
        print("abc")
        le = LabelEncoder()
        self.dataset[column_name] = le.fit_transform(self.dataset[column_name])

    # one-hot encoding
    def one_hot_encoding(self, column_name):
        le2 = LabelEncoder()
        le_encoded = le2.fit_transform(self.dataset[column_name])
        if len(le2.classes_) > 2:
            one_hot_encoder = OneHotEncoder(sparse=False)
            one_hot_encoded = one_hot_encoder.fit_transform(le_encoded.reshape(-1,1))
            df_encoded = pd.DataFrame(one_hot_encoded)
            self.dataset = pd.merge(self.dataset, df_encoded, left_index=True, right_index=True, how='inner')
            del self.dataset[column_name]
        else:
            self.dataset[column_name] = le_encoded
        
    # deleting columns
    def drop_columns(self, column_name):
        del self.dataset[column_name]

    # setting the target variable
    def setting_target_variable(self, target_column):
        self.y = self.dataset[target_column]
        del self.dataset[target_column]
        self.X = self.dataset

    # transforming target variable based on shipment days
    def transform_target_variable_function(self, x):
        if x in [0,1]:
            return 0
        elif x in [2,3]:
            return 1
        else:
            return 2

    # transforming the target variable
    def transform_target_variable(self):
        self.dataset["Days for shipping (real)"] = self.dataset["Days for shipping (real)"].apply(lambda x: self.transform_target_variable_function(x))

    # splitting the dataset
    def splitting_the_dataset(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, stratify=self.y)

    # returning the train-test split dataset
    def return_train_test_split_dataset(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    # return features and 
    def return_train_target(self):
        return self.X.values, self.y.values
    
    # returning the preprocessed dataset
    def return_dataset(self):
        return self.dataset








