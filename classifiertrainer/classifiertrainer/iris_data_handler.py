#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 00:04:19 2022

@author: priyankamocherla

Class to load sklearns Iris dataset and process it before training models
"""

from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

class IrisDataHandler:

    def __init__(self, seed=None):
        self.seed = seed
        
        self.expected_features = {  'SepalLengthCM': float,
                                    'SepalWidthCM': float, 
                                    'PetalLengthCM': float, 
                                    'PetalWidthCM': float}
        self.target = {'Target' : int}
        self.data = self.load_data_from_sklearn()
        

    def load_data_from_sklearn(self):
        print('Loading iris dataset...')
        iris_data = datasets.load_iris()
        iris_df = pd.DataFrame(iris_data['data'], columns=list(self.expected_features.keys()))
        iris_df['Target'] = iris_data['target']
        
        print(f'{len(iris_df)} samples loaded.')
        
        return iris_df

    
    def split_data(self, test_size=0.4, shuffle=True, normalize_data=True):
        
        print(f'Test data split: {test_size}')

        X_train, X_test, y_train, y_test = train_test_split(self.data[list(self.expected_features.keys())], 
                                                            self.data['Target'],
                                                            test_size=test_size,
                                                            random_state=self.seed, 
                                                            shuffle=shuffle)
        
        print(f'{len(X_train)} train samples, {len(X_test)} test samples\n')
        
        if normalize_data:
            X_train = normalize(X_train)
            X_test = normalize(X_test)
        
        return  X_train, X_test, y_train, y_test
    
