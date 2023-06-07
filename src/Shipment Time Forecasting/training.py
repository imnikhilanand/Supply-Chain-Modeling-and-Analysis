# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 16:25:59 2023

@author: Nikhil
"""

# importing the libraries
import pandas as pd
import numpy as np
from preprocessing import Preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# creating class for training
class Training(RandomForestClassifier, LogisticRegression):
    
    # creating the constructor
    def __init__(self, X_train, y_train, X_test, y_test, classifier):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        if classifier == 'RandomForestClassfier':
            self.model = RandomForestClassifier(max_depth=100, 
                                                n_estimators=500,
                                                min_samples_split=5,
                                                min_samples_leaf=3,
                                                max_features="sqrt",
                                                bootstrap="True"
                                                )
        elif classifier == "LogisticRegression":
            self.model = LogisticRegression()
        self.y_predict = None
        self.y_test = y_test
        
    # cross-validation
    def train_cross_val(self):
        scores = cross_val_score(self.model, self.X_train, self.y_train)
        return scores
    
    # training the model
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        self.y_predict = self.model.predict(self.X_test)

    # returning prediction score
    def return_predict(self):
        return self.y_predict
    
    # returning accuracy score
    def return_accuracy_score(self):
        return accuracy_score(self.y_predict, self.y_test)
    
    # returning confusion matric
    def return_confusion_matrix(self):
        return confusion_matrix(self.y_test, self.y_predict)
    
    # returning metrics report
    def return_metric_report(self):
        return precision_recall_fscore_support(self.y_test, self.y_predict)
    
    def return_feature_importance(self):
        if classifier == 'RandomForestClassifier':
            return self.model.feature_importances_
        else:
            return "Not Applicable"


# loading the dataset
dataset = pd.read_csv("../../data/DataCoSupplyChainDataset.csv", encoding='latin')
org_dataset = dataset.copy()

# calling the object
pr = Preprocessing(dataset)

# drop irrelevant columns
pr.drop_columns('Days for shipment (scheduled)')
pr.drop_columns('Delivery Status')
pr.drop_columns('Late_delivery_risk')
pr.drop_columns('Category Name')
pr.drop_columns('Customer City')
pr.drop_columns('Customer Country')
pr.drop_columns('Customer Email')
pr.drop_columns('Customer Fname')
pr.drop_columns('Customer Id')
pr.drop_columns('Customer Lname')
pr.drop_columns('Customer Password')
pr.drop_columns('Customer Segment')
pr.drop_columns('Customer State')
pr.drop_columns('Customer Street')
pr.drop_columns('Customer Zipcode')
pr.drop_columns('Department Name')
pr.drop_columns('Latitude')
pr.drop_columns('Longitude')
pr.drop_columns('Order Customer Id')
pr.drop_columns('order date (DateOrders)')
pr.drop_columns('Order Id')
pr.drop_columns('Order Item Cardprod Id')
pr.drop_columns('Order Item Discount')
pr.drop_columns('Order Item Discount Rate')
pr.drop_columns('Order Item Id')
pr.drop_columns('Order Item Product Price')
pr.drop_columns('Order Item Profit Ratio')
pr.drop_columns('Sales')
pr.drop_columns('Order Item Total')
pr.drop_columns('Order Profit Per Order')
pr.drop_columns('Order Status')
pr.drop_columns('Order Zipcode')
pr.drop_columns('Product Card Id')
pr.drop_columns('Product Category Id')
pr.drop_columns('Product Description')
pr.drop_columns('Product Image')
pr.drop_columns('Product Name')
pr.drop_columns('Product Price')
pr.drop_columns('Product Status')
pr.drop_columns('shipping date (DateOrders)')

# label encoding relevant columns
pr.label_encode('Type')
pr.label_encode('Market')
pr.label_encode('Order City')
pr.label_encode('Order Country')
pr.label_encode('Order Region')
pr.label_encode('Order State')
pr.label_encode('Shipping Mode')

# transform target variable
pr.transform_target_variable()

# setting the target columns
pr.setting_target_variable('Days for shipping (real)')

# splitting the dataset
pr.splitting_the_dataset()
    
# returning the dataset
X_train, X_test, y_train, y_test = pr.return_train_test_split_dataset()

# calling object for model training
mt = Training(X_train, y_train, X_test, y_test, 'RandomForestClassfier')

# training the model
mt.train_model()

# getting the accuracy score
mt.return_accuracy_score()

# getting the report
mt.return_metric_report()

# confusion matrix
mt.return_confusion_matrix()