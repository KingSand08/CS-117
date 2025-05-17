import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score

from format_data import format_data as data
from k_NN_training_final_project import knn_builder

from ml_settings import seed

np.random.seed(seed)


def build_ensemble_voting(layerSizes, act_func, solver_func, max_epochs, given_batch_size, patience, val_per, early_stop_per, n_neighbors_size, weights_size, p_size, verboseness):
    knn_model = knn_builder(n_neighbors_size, weights_size, p_size)
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


    #? Validation Accuracy
    val_preds = v.predict(validation_inputs)
    val_acc = accuracy_score(validation_targets, val_preds)
    print(f"Validation accuracy: {val_acc:.2f}")

    #? Test Accuracy
    test_preds = v.predict(test_inputs)
    test_acc = accuracy_score(test_targets, test_preds)
    print(f"Test accuracy:       {test_acc:.2f}")

    #? F1 Stoke Score
    test_f1_stroke = f1_score(test_targets, test_preds, pos_label=1)
    print(f"F1-score (Stroke):   {test_f1_stroke:.2f}")


    return test_f1_stroke
    
def build_ensemble_bagging_dl(estimator_num, layerSizes, act_func, solver_func, max_epochs, given_batch_size, patience, val_per, early_stop_per, verboseness) :
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


    #? Validation Accuracy
    val_preds = b.predict(validation_inputs)
    val_acc = accuracy_score(validation_targets, val_preds)
    print(f"Validation accuracy: {val_acc:.2f}")

    #? Test Accuracy
    test_preds = b.predict(test_inputs)
    test_acc = accuracy_score(test_targets, test_preds)
    print(f"Test accuracy:       {test_acc:.2f}")

    #? F1 Stoke Score
    test_f1_stroke = f1_score(test_targets, test_preds, pos_label=1)
    print(f"F1-score (Stroke):   {test_f1_stroke:.4f}")
    
    
    return test_f1_stroke
    
def build_ensemble_bagging_knn(estimator_num, n_neighbors_size, weights_size, p_size):
    knn_model = knn_builder(n_neighbors_size, weights_size, p_size)
    
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


    #? Validation Accuracy
    val_preds = b.predict(validation_inputs)
    val_acc = accuracy_score(validation_targets, val_preds)
    print(f"Validation accuracy: {val_acc:.2f}")

    #? Test Accuracy
    test_preds = b.predict(test_inputs)
    test_acc = accuracy_score(test_targets, test_preds)
    print(f"Test accuracy:       {test_acc:.2f}")

    #? F1 Stoke Score
    test_f1_stroke = f1_score(test_targets, test_preds, pos_label=1)
    print(f"F1-score (Stroke):   {test_f1_stroke:.4f}")

    # print(accuracy)
    return test_f1_stroke