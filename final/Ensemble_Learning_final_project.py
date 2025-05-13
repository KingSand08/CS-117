import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

from format_data import format_data as data
from k_NN_training_final_project import knn_builder

from ml_settings import seed


def build_ensemble_voting(layerSizes, act_func, solver_func, max_epochs, given_batch_size, patience, val_per, verboseness, early_stop_per):
    knn_model = knn_builder()
    dl_model = MLPClassifier(
        hidden_layer_sizes=layerSizes,
        activation=act_func,
        solver=solver_func,
        max_iter=max_epochs,
        batch_size=given_batch_size,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=patience,
        validation_fraction=val_per,
        alpha=early_stop_per,
        verbose=verboseness
    )
    
    ## create a voting classifier 
    model_list = [('knn',knn_model),('dl',dl_model)]
    
    v = VotingClassifier(
        estimators = model_list , 
        n_jobs=-1
    )
    
    return v


def run_ensemble_voting(v):
    train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets = data()

    # train the voting classifier 
    v.fit(train_inputs,train_targets)


    # make predictions 
    predictions = v.predict(test_inputs)


    # get model accuracy 
    accuracy = accuracy_score(test_targets,predictions)

    # print(accuracy)
    return accuracy
    
def build_ensemble_bagging_dl(estimator_num, layerSizes, act_func, solver_func, max_epochs, given_batch_size, patience, val_per, verboseness, early_stop_per) :
    dl_model = MLPClassifier(
        hidden_layer_sizes=layerSizes,
        activation=act_func,
        solver=solver_func,
        max_iter=max_epochs,
        batch_size=given_batch_size,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=patience,
        validation_fraction=val_per,
        alpha=early_stop_per,
        verbose=verboseness
    )
    
    b = BaggingClassifier(
        estimator = dl_model, 
        n_estimators= estimator_num,
        n_jobs=-1,
        random_state=seed
    )
    
    return b


def run_ensemble_bagging_dl(b):
    train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets = data()
    
    # train the voting classifier 
    b.fit(train_inputs,train_targets)


    # make predictions 
    predictions = b.predict(test_inputs)


    # get model accuracy 
    accuracy = accuracy_score(test_targets,predictions)

    # print(accuracy)
    return accuracy
    
def build_ensemble_bagging_knn(estimator_num):
    knn_model = knn_builder()
    
    b = BaggingClassifier(
        estimator = knn_model, 
        n_estimators=estimator_num,
        n_jobs=-1,
        random_state=seed
    )
    
    return b
    
def run_ensemble_bagging_knn(b):
    train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets = data()
    
    # train the voting classifier 
    b.fit(train_inputs,train_targets)


    # make predictions 
    predictions = b.predict(test_inputs)


    # get model accuracy 
    accuracy = accuracy_score(test_targets,predictions)

    # print(accuracy)
    return accuracy