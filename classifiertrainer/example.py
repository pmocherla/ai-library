#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 00:10:11 2022

@author: priyankamocherla

"""

from classifiertrainer.iris_data_handler import IrisDataHandler
from classifiertrainer.classifier_trainer import KNNTrainer
import matplotlib.pyplot as plt


def train_and_eval_model(params, X_train, X_test, y_train, y_test):
    knn = KNNTrainer(params)
    knn.train(X_train, y_train)
    results = knn.evaluate(X_test, y_test)
    
    return results


def plot_results(results_list):
    x = list(int(i) for i in results_list.keys())
    acc = [results_list[i]['accuracy'] for i in x]
    pre = [results_list[i]['precision'] for i in x]
    rec = [results_list[i]['recall'] for i in x]
    plt.figure()
    plt.plot(x, acc, label='Accuracy')
    plt.plot(x, pre, label='Precision')
    plt.plot(x, rec, label='Recall')
    plt.ylabel('Score')
    plt.xlabel('k neighbours')
    plt.legend()
    plt.show()
    return None

def example_training_func():
    seed = 8
    iris_data = IrisDataHandler(seed=seed)
    X_train, X_test, y_train, y_test = iris_data.split_data()
    
    test_params = [{'n_neighbors' : i} for i in range(1,21)]
    results = {}
    
    print('###### TRAINING ######')
    for i, params in enumerate(test_params):
        print(f'\nTraining model {i}: {params}')
        _, results_dict = train_and_eval_model(params, X_train, X_test, y_train, y_test)
        results[i] = results_dict
    
    plot_results(results)

if __name__ == "__main__":
    example_training_func()
    

    