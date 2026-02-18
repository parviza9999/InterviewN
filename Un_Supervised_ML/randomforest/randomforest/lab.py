import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

################################################################################
#                                                                              #
#   Copyright (c) 2026 Interview Node.                                         #
#   All Rights Reserved.                                                       #
#                                                                              #
#   This software is the confidential and proprietary information of           #
#   Interview Node ("Confidential Information").                               #
#                                                                              #
################################################################################



# ==========================================
# LAB SETUP & VISUALIZATION HELPERS
# ==========================================
try:
    with open('solutions_rf.pkl', 'rb') as f:
        data = pickle.load(f)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        TARGET_DT_ACC = data['target_dt_acc']
        TARGET_RF_ACC = data['target_rf_acc']
        CORRECT_TOP_F = data['top_features']
    print(f"Data Loaded. Target DT Score: {TARGET_DT_ACC:.3f} | Target RF Score: {TARGET_RF_ACC:.3f}")
except FileNotFoundError:
    print("CRITICAL ERROR: 'solutions_rf.pkl' not found.")
    exit()

def plot_overfitting(train_acc, test_acc):
    """Visualizes the gap between training and testing (The Overfitting Gap)."""
    labels = ['Train Score', 'Test Score']
    values = [train_acc, test_acc]
    colors = ['#ff9999', '#66b3ff'] # Red (danger/train), Blue (safe/test)
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=colors, alpha=0.8)
    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy')
    plt.title('Task 1 Result: The Overfitting Gap')
    
    # Add text on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', fontweight='bold')
    
    plt.axhline(y=TARGET_DT_ACC, color='green', linestyle='--', label='Target Goal')
    plt.legend()
    plt.show()

def plot_improvement(dt_score, rf_score):
    """Visualizes the improvement of Forest over Single Tree."""
    labels = ['Single Tree', 'Random Forest']
    values = [dt_score, rf_score]
    colors = ['gray', 'green']
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=colors, alpha=0.8)
    plt.ylim(0, 1.1)
    plt.ylabel('Test Accuracy')
    plt.title('Task 2 Result: Single Tree vs. Forest')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', fontweight='bold')

    plt.show()

# ==========================================
# STUDENT SECTION
# ==========================================

print("\n--- TASK 1: Fixing Overfitting ---")
print("Instructions: Run the code. A perfect Train Score (1.0) with a low Test Score means Overfitting.")
print("The model is memorizing the noise in the data.")


# [TODO] TASK 1: Limit the Tree Depth
# CURRENT STATUS: max_depth=None (Unlimited depth -> Overfitting)
# YOUR GOAL: Change max_depth to a small integer (e.g., 3, 5, or 7) to close the gap.
dt_model = DecisionTreeClassifier(
    max_depth=None,       # <--- CHANGE THIS to an integer (e.g. 5)
    random_state=42
)
dt_model.fit(X_train, y_train)

dt_train_score = dt_model.score(X_train, y_train)
dt_test_score = dt_model.score(X_test, y_test)

# VISUALIZE TASK 1
plot_overfitting(dt_train_score, dt_test_score)

# Grading Task 1
if abs(dt_test_score - TARGET_DT_ACC) <= 0.02:
    print(f"✅ Task 1: PASS (Score: {dt_test_score:.3f})")
    score1 = 1
else:
    print(f"❌ Task 1: FAIL (Got {dt_test_score:.3f}, Expected ~{TARGET_DT_ACC:.3f})")
    score1 = 0


print("\n--- TASK 2: The Power of Ensembling ---")
print("Instructions: A single tree is unstable. A Random Forest uses 'Bagging' to vote on the best result.")


# [TODO] TASK 2: Increase Estimators
# CURRENT STATUS: n_estimators=1 (Too few trees)
# YOUR GOAL: Increase n_estimators to 100.
rf_model = RandomForestClassifier(
    n_estimators=1,       # <--- CHANGE THIS to 100
    max_depth=5,          # Keeping depth constrained
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_test_score = rf_model.score(X_test, y_test)

# VISUALIZE TASK 2
plot_improvement(dt_test_score, rf_test_score)

# Grading Task 2
if abs(rf_test_score - TARGET_RF_ACC) <= 0.02:
    print(f"✅ Task 2: PASS (Score: {rf_test_score:.3f})")
    score2 = 1
else:
    print(f"❌ Task 2: FAIL (Got {rf_test_score:.3f}, Expected ~{TARGET_RF_ACC:.3f})")
    score2 = 0


print("\n--- TASK 3: Feature Importance ---")
print("Instructions: The Random Forest can tell you which features are just noise.")
print("We have 20 features (indices 0-19). Only 5 are real signal. The rest are junk.")

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# VISUALIZE TASK 3 (Existing visualization logic)
plt.figure(figsize=(10, 4))
plt.title("Task 3 Result: Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center", color='purple')
plt.xticks(range(X_train.shape[1]), indices)
plt.xlabel("Feature Index (Sorted by Importance)")
plt.ylabel("Importance Score")
plt.show()

# [TODO] TASK 3: Identify Top Features
# Look at the purple chart. Which 5 feature indices have the tallest bars?
# Enter them in the list below.
top_5_submission = [99, 99, 99, 99, 99]  # <--- REPLACE these numbers with the top 5 indices

# Grading Task 3
if set(top_5_submission) == CORRECT_TOP_F:
    print("✅ Task 3: PASS (You found the signal)")
    score3 = 1
else:
    print(f"❌ Task 3: FAIL (Check the graph again. You submitted {set(top_5_submission)})")
    score3 = 0

# ==========================================
# FINAL SCORE
# ==========================================
total = score1 + score2 + score3
print(f"\nFINAL SCORE: {total}/3")
if total == 3:
    print("🎉 LAB COMPLETE! You have mastered Random Forests.")