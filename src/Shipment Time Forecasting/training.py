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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# creating class for training
class Training(RandomForestClassifier, LogisticRegression):
    
    # creating the constructor
    def __init__(self, X_train, y_train, X_test, y_test, classifier):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.classifier = classifier
        self.y_predict = None
        self.y_test = y_test
        self.pca = None
        if self.classifier == 'RandomForestClassfier':
            self.model = RandomForestClassifier(max_depth=100, 
                                                n_estimators=500,
                                                min_samples_split=5,
                                                min_samples_leaf=3,
                                                max_features="sqrt",
                                                bootstrap="True"
                                                )
        elif self.classifier == "LogisticRegression":
            self.model = LogisticRegression()
        elif self.classifier == "Gaussian Naive Bayes":
            self.model = GaussianNB()
        elif self.classifier == "KNN":
            self.model = KNeighborsClassifier(n_neighbors=3)
        elif self.classifier == "xgboost":
            params = {'objective': 'multi:softmax','num_class': 3,'eval_metric': 'merror'}
            dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
            dtest = xgb.DMatrix(self.X_test)
            self.model = xgb.train(params, dtrain)
            self.y_predict = self.model.predict(dtest)
        
        
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
        print(self.classifier)
        if self.classifier == 'RandomForestClassfier':
            return self.model.feature_importances_
        else:
            return "Not Applicable"

    # checking PCA
    def check_PCA(self, dimensions):
        pca = PCA(n_components = dimensions)
        X_train_reduced = pca.fit_transform(self.X_train)
        X_test_reduced = pca.transform(self.X_test)
        return pca.explained_variance_ratio_

    # implementing PCA
    def reduce_PCA(self, dimensions):
        pca = PCA(n_components = dimensions)
        self.X_train = pca.fit_transform(self.X_train)
        self.X_test = pca.transform(self.X_test)
        

''' LOADING THE DATASET (Common for all the algorithms) '''

# loading the dataset
dataset = pd.read_csv("../../data/DataCoSupplyChainDataset.csv", encoding='latin')
org_dataset = dataset.copy()

# calling the object
pr = Preprocessing(dataset)

###############################################################################
###############################################################################
''' FOR ALGORITHMS THAT DO NOT REQUIRES LABEL ENCODING '''
###############################################################################
###############################################################################

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
#mt = Training(X_train, y_train, X_test, y_test, 'RandomForestClassfier')
#mt = Training(X_train, y_train, X_test, y_test, 'Gaussian Naive Bayes')
#mt = Training(X_train, y_train, X_test, y_test, 'KNN')
mt = Training(X_train, y_train, X_test, y_test, 'xgboost')

# training the model
mt.train_model()

# getting the accuracy score
mt.return_accuracy_score()

# getting the report
mt.return_metric_report()

# confusion matrix
mt.return_confusion_matrix()

# feature improtance of features
mt.return_feature_importance()

# feature importance
columns_in_table = pr.return_dataset().columns
importance_of_features = mt.return_feature_importance()

# importance of features
importance_of_features = pd.DataFrame({'features':columns_in_table, 'feature_importance':importance_of_features})
importance_of_features = importance_of_features.sort_values(['feature_importance'], ascending=False)

# featues plot
sns.barplot(data=importance_of_features, x = "features",y = "feature_importance")


###############################################################################
###############################################################################
''' FOR ALGORITHMS THAT REQUIRES LABEL ENCODING '''
###############################################################################
###############################################################################

# drop irrelvant columns
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

# additional deletinon of few columns for algorithms that requires one hot encoding
pr.drop_columns('Order City')
pr.drop_columns('Order Country')
pr.drop_columns('Order State')
pr.drop_columns('Order Region')

# one hot encoding features
pr.one_hot_encoding('Type')
pr.one_hot_encoding('Market')
pr.one_hot_encoding('Category Id')
pr.one_hot_encoding('Department Id')
pr.one_hot_encoding('Shipping Mode')

# return_dataset 
new_dataset = pr.return_dataset()

# transform target variable
pr.transform_target_variable()

# setting the target columns
pr.setting_target_variable('Days for shipping (real)')

# splitting the dataset
pr.splitting_the_dataset()
    
# returning the dataset
X_train, X_test, y_train, y_test = pr.return_train_test_split_dataset()

# calling object for model training
mt = Training(X_train, y_train, X_test, y_test, 'LogisticRegression')

# implement PCA
mt.check_PCA(10)

# implement PCA
mt.reduce_PCA(10)

# training the model
mt.train_model()

# getting the accuracy score
mt.return_accuracy_score()

# getting the report
mt.return_metric_report()

# confusion matrix
mt.return_confusion_matrix()




