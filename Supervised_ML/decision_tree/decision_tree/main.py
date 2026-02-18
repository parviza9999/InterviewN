#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
#                                                                              #
#   Copyright (c) 2026 Interview Node.                                         #
#   All Rights Reserved.                                                       #
#                                                                              #
#   This software is the confidential and proprietary information of           #
#   Interview Node ("Confidential Information").                               #
#                                                                              #
################################################################################

"""
LAB: Decision Trees
OBJECTIVE: Train, Visualize, and Prune a Decision Tree.
NOTE: Visualization code is provided. You must train the models that feed the plots.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

def main():
    # ==========================================
    # PHASE 1: Data Preparation
    # ==========================================
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names, class_names = iris.feature_names, list(iris.target_names)

    # [TASK 1] Split Data
    # TODO: Split X, y into training/testing (test_size=0.3, random_state=42)
    # X_train, X_test, y_train, y_test = ...

    # --- DUMMY DATA FOR TASK 1 ---
    # Overwrite these with your actual split logic above
    X_train, y_train = X, y 
    # -----------------------------

    # ==========================================
    # PHASE 2: Full Tree Training
    # ==========================================

    # [TASK 2] Train Full Tree
    # TODO: Initialize DecisionTreeClassifier(criterion='gini') and fit on X_train.
    # Name your model 'clf_full'.
    
    # Dummy model to ensure plot works before you finish the task
    clf_full = DecisionTreeClassifier(max_depth=1).fit(X_train, y_train) 
    
    # YOUR CODE HERE:
    # clf_full = ...
    # clf_full.fit(...)

    # --- PLOT SETUP (DO NOT EDIT) ---
    # Visualizes whatever model is stored in 'clf_full'
    print("Visualizing Full Tree...")
    plt.figure(figsize=(12, 8))
    plot_tree(clf_full, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    plt.title("Full Decision Tree")
    plt.show()
    # --------------------------------

    # ==========================================
    # PHASE 3: Pruning
    # ==========================================
    
    # [TASK 3] Train Pruned Tree
    # TODO: Train a new model 'clf_pruned' with max_depth=3.
    
    # Dummy model for plotting
    clf_pruned = DecisionTreeClassifier(max_depth=1).fit(X_train, y_train)

    # YOUR CODE HERE:
    # clf_pruned = ...
    # clf_pruned.fit(...)

    # --- PLOT SETUP (DO NOT EDIT) ---
    print("Visualizing Pruned Tree...")
    plt.figure(figsize=(10, 8))
    plot_tree(clf_pruned, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    plt.title("Pruned Decision Tree (Depth=3)")
    plt.show()
    # --------------------------------

if __name__ == "__main__":
    main()