# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:46:00 2022

@author: priyanka.mocherla

Classifier abstract class for training sklearn classifier models. Current models supported:
    - KNN
"""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier

class ClassifierTrainer:
    
    def __init__(self, params):
        self.model = None
        
    def train(self, train_data, train_labels):
        pass
    
    def evaluate(self, test_data, test_labels):
        pass
    
    @staticmethod
    def calculate_evaluation_metrics(test_pred, test_labels):
        metrics = {'accuracy' : None,
                   }
        metrics['accuracy'] = accuracy_score(test_labels, test_pred)
        metrics['precision'], metrics['recall'], metrics['fscore'], metrics['support'] = precision_recall_fscore_support(test_labels, test_pred, average='macro')
    
        return metrics
        
    def get_model(self):
        return self.model
    
    

class KNNTrainer(ClassifierTrainer):
    """Creates a KNN model with given params to train and evaluate"""
    
    def __init__(self, params):
        self.model = KNeighborsClassifier(**params)
        
    
    def train(self, train_data, train_labels):
        
        assert(len(train_data) > 0)
        assert(len(train_data) == len(train_labels))
        
        self.model.fit(train_data, train_labels)
        
        return self.model
    
    def evaluate(self, test_data, test_labels):
        
        if not self.model:
            raise Exception('No model found. Train model using KNNModel.train_model first.')
            
        assert(len(test_data) > 0)
        assert(len(test_data) == len(test_labels))
            
        test_pred = self.model.predict(test_data)
        metrics = self.calculate_evaluation_metrics(test_pred, test_labels)
        print(f'Test Accuracy -> k = {self.model.n_neighbors}: {metrics["accuracy"]}')
        
        return test_pred, metrics
    
    
