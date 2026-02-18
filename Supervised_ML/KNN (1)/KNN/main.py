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
LAB: K-Nearest Neighbors (K-NN) Implementation & Optimization
OBJECTIVE: Understand feature scaling, distance metrics, and hyperparameter tuning.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# [TASK 1] Import necessary modules from Scikit-Learn
# TODO: Import train_test_split, StandardScaler, and KNeighborsClassifier
# from sklearn.model_selection import ...
# from sklearn.preprocessing import ...
# from sklearn.neighbors import ...


def main():
    # ==========================================
    # PHASE 1: Data Loading & Visualization
    # ==========================================
    print("Loading Data...")
    
    # We will use the Iris dataset for this lab
    # The Iris dataset is a classic, small dataset in machine learning featuring 150 flower 
    # samples (50 each of Iris setosa, versicolor, and virginica) with four measurements (sepal 
    #length/width, petal length/width) to classify the species
    iris = load_iris()
    X = iris.data[:, :2]  # We only take the first two features for visualization purposes
    y = iris.target

    # [TASK 2] Visualize the Data
    # TODO: Create a scatter plot of the data. 
    # Hint: Use plt.scatter. Color the points by 'y' (the target class).
    # Label the x-axis "Sepal Length" and y-axis "Sepal Width".
    # Important: Save the  plot as scatterplot.png
    
    plt.figure(figsize=(8, 6))
    # <YOUR CODE HERE>
    plt.title("Iris Data Distribution (First 2 Features)")
    plt.show()

    # ==========================================
    # PHASE 2: Preprocessing
    # ==========================================
    
    # [TASK 3] Split the Data
    # TODO: Split X and y into training and testing sets.
    # Use a test_size of 0.3 (30%) and random_state=42 for reproducibility.
    
    # X_train, X_test, y_train, y_test = ... 

    # [TASK 4] Feature Scaling (Crucial for K-NN)
    # TODO: K-NN is distance-based, so features must be on the same scale.
    # 1. Initialize a StandardScaler.
    # 2. Fit the scaler on X_train ONLY (to avoid data leakage).
    # 3. Transform both X_train and X_test using the scaler.
    
    # scaler = ...
    # X_train = ...
    # X_test = ...

    # ==========================================
    # PHASE 3: The Math of Distance
    # ==========================================
    
    # [TASK 5] Euclidean Distance Function
    # TODO: Write a function that calculates the Euclidean distance between two numpy arrays (points).
    # Formula: sqrt(sum((p1 - p2)^2))
    # This is just a helper task to ensure you understand the geometry.
    
    def euclidean_distance(point1, point2):
        pass # <YOUR CODE HERE>

    # Test your function (Do not modify this part)
    p1 = np.array([1, 2])
    p2 = np.array([4, 6])
    dist = euclidean_distance(p1, p2)
    print(f"\n[Math Check] Distance between (1,2) and (4,6) should be 5.0. You got: {dist}")

    # ==========================================
    # PHASE 4: Model Training & Prediction
    # ==========================================

    # [TASK 6] Train the K-NN Classifier
    # TODO: Initialize the KNeighborsClassifier with n_neighbors=3.
    # Fit the model on your scaled training data.
    
    # knn = ...
    # <YOUR CODE HERE to fit model>

    # [TASK 7] Evaluate the Model
    # TODO: Predict labels for X_test and print the accuracy score.
    
    # y_pred = ...
    # accuracy = ...
    # print(f"Model Accuracy with K=3: {accuracy:.2f}")

    # ==========================================
    # PHASE 5: Optimization 
    # ==========================================
    
    # [TASK 8] Find the Optimal K
    # TODO: 
    # 1. Create a loop that runs from K=1 to K=40.
    # 2. In each iteration, train a KNN model with that K.
    # 3. Calculate the error rate (mean(y_pred != y_test)) and append it to error_rate list.Do not change the name of error_rate list.
    # 4. Observe the plot and answer the question in assignment.txt
    error_rate = []
    for i in range(1, 40):
        error_rate.append(0)
    
    # for i in range(1, 40):
    #     <YOUR CODE HERE>
    
    # Plot the Elbow Graph 
    # [TASK 9] Save the graph as elbow.png
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    print("Displaying Elbow Plot...")
    plt.show()

if __name__ == "__main__":
    main()