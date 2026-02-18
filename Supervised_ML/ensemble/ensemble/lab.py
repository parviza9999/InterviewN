import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# ==========================================
# LAB SETUP
# ==========================================
try:
    with open('solutions_bag_boost.pkl', 'rb') as f:
        pkg = pickle.load(f)
        X_train, X_test, y_train, y_test = pkg['data']
        TARGETS = pkg['targets']
        QUIZ_KEY = pkg['quiz_key']
    print("✅ Lab Data Loaded. Dataset: Nested Circles")
except FileNotFoundError:
    print("❌ ERROR: 'solutions_bag_boost.pkl' not found.")
    exit()

def plot_decision_boundaries(clf, title):
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k', cmap='RdYlBu')
    plt.title(title)
    plt.show()

# ==========================================
# TASK 1: THE WEAKNESS OF A SINGLE TREE
# ==========================================
print("\n--- TASK 1: Single Decision Tree ---")
# [TODO] Create a default DecisionTreeClassifier with random_state=42
dt_clf = DecisionTreeClassifier(random_state=4)  # <--- CHANGE TO 42 and check the accuracy
dt_clf.fit(X_train, y_train)
dt_acc = dt_clf.score(X_test, y_test)

print(f"Single Tree Test Accuracy: {dt_acc:.3f}")
plot_decision_boundaries(dt_clf, "Task 1: Single Decision Tree (Overfit?)")

# ==========================================
# TASK 2: BAGGING (RANDOM FOREST)
# ==========================================
print("\n--- TASK 2: Bagging (Random Forest) ---")
print("Goal: Reduce variance by averaging 100 deep trees.")


# [TODO] Create a RandomForestClassifier. 
# Set n_estimators=100, max_depth=5, and random_state=42.
rf_clf = RandomForestClassifier(
    n_estimators=1,   # <--- CHANGE TO 100
    max_depth=1,      # <--- CHANGE TO 5
    random_state=42
)
rf_clf.fit(X_train, y_train)
rf_acc = rf_clf.score(X_test, y_test)

plot_decision_boundaries(rf_clf, "Task 2: Random Forest (Bagging)")
score2 = 1 if abs(rf_acc - TARGETS['rf']) < 0.01 else 0

# ==========================================
# TASK 3: BOOSTING (ADABOOST)
# ==========================================
print("\n--- TASK 3: Boosting (AdaBoost) ---")
print("Goal: Turn weak 'stumps' into a strong learner by focusing on errors.")


# [TODO] Create an AdaBoostClassifier.
# Use a DecisionTreeClassifier with max_depth=1 as the base estimator.
# Set n_estimators=100, learning_rate=0.5, and random_state=42.
ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=10), # <--- CHANGE depth to 1
    n_estimators=10,                                # <--- CHANGE to 100
    learning_rate=0.5,
    random_state=42
)
ada_clf.fit(X_train, y_train)
ada_acc = ada_clf.score(X_test, y_test)

plot_decision_boundaries(ada_clf, "Task 3: AdaBoost (Boosting)")
score3 = 1 if abs(ada_acc - TARGETS['ada']) < 0.01 else 0

# ==========================================
# TASK 4: ANALYZING SENSITIVITY
# ==========================================
print("\n--- TASK 4: Learning Rate in Boosting ---")
# [TODO] What happens if the learning_rate is too high (e.g., 2.0)? 
# Predict if the accuracy will [Increase] or [Decrease].
# Set the variable below to your string choice.
ada_sensitivity = "Decrease" # Update after testing manually if you wish!
score4 = 1 # Self-graded based on exploration

# ==========================================
# TASK 5: ENSEMBLE COMPARISON
# ==========================================
print("\n--- TASK 5: Performance Recap ---")
# No code needed, just run to see the comparison
print(f"Tree Accuracy: {dt_acc:.3f}")
print(f"RF Accuracy:   {rf_acc:.3f}")
print(f"Ada Accuracy:  {ada_acc:.3f}")
score5 = 1

# ==========================================
# CONCEPTUAL QUIZ
# ==========================================
print("\n--- QUIZ: BAGGING VS BOOSTING ---")
# [TODO] Replace None with True or False
student_quiz = [
    None, # Q1: Bagging reduces variance by averaging multiple independent trees.
    None, # Q2: In Boosting, trees are grown in parallel independently of each other.
    None, # Q3: AdaBoost gives more weight to samples misclassified by previous trees.
    None, # Q4: Random Forest uses a subset of features at each split to increase diversity.
    None  # Q5: Boosting is generally less prone to overfitting than Bagging if you use too many trees.
]

quiz_points = 0
if None not in student_quiz:
    for i, (ans, correct) in enumerate(zip(student_quiz, QUIZ_KEY)):
        if ans == correct: quiz_points += 1
        else: print(f"🚩 Q{i+1} Incorrect.")

# ==========================================
# FINAL RESULTS
# ==========================================
print("\n" + "="*40)
print(f"LAB SCORE: {score2 + score3 + score4 + score5}/4 Coding Tasks")
print(f"QUIZ SCORE: {quiz_points}/5")
print("="*40)