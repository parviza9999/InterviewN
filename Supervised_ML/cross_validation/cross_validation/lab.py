import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

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
# LAB SETUP (DO NOT EDIT)
# ==========================================
try:
    with open('solutions_cv.pkl', 'rb') as f:
        data = pickle.load(f)
        X, y = data['X'], data['y']
        LUCKY_SCORE = data['lucky_score']
        TARGET_CV_MEAN = data['target_cv_mean']
        TARGET_CV_SCORES = data['target_cv_scores']
        QUIZ_KEY = data['quiz_key']
    print("✅ Lab environment initialized successfully.")
except FileNotFoundError:
    print("❌ ERROR: 'solutions_cv.pkl' not found. Ensure you ran the instructor script.")
    exit()

def plot_cv_variance(cv_scores, lucky_score):
    """Visualizes how different folds compare to the 'lucky' single split."""
    plt.figure(figsize=(8, 5))
    folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
    plt.bar(folds, cv_scores, color='skyblue', label='K-Fold Scores')
    plt.axhline(y=np.mean(cv_scores), color='blue', linestyle='--', label='CV Mean')
    plt.axhline(y=lucky_score, color='red', linestyle=':', label='Single Lucky Split')
    plt.ylim(min(cv_scores)-0.05, 1.0)
    plt.ylabel('Accuracy Score')
    plt.title('Task 2: Accuracy Variance Across Folds')
    plt.legend(loc='lower right')
    plt.show()

# ==========================================
# TASK 1: THE ILLUSION OF THE SINGLE SPLIT
# ==========================================
print("\n--- TASK 1: The Illusion of the Single Split ---")
print(f"A standard train/test split yielded a score of: {LUCKY_SCORE:.4f}")
print("This score might be 'lucky' based on how the data was shuffled.")

# No code to change here; proceed to Task 2 to verify this score's reliability.
score1 = 1 

# ==========================================
# TASK 2: K-FOLD CROSS VALIDATION
# ==========================================
print("\n--- TASK 2: Implementing K-Fold Cross-Validation ---")
print("Instructions: Change the 'cv' parameter to 5 to perform a robust evaluation.")



model = LogisticRegression(max_iter=10000)

# [TODO] TASK 2: Change cv=2 to cv=5
cv_results = cross_val_score(
    estimator=model, 
    X=X, 
    y=y, 
    cv=2  # <--- CHANGE THIS TO 5
)

student_cv_mean = np.mean(cv_results)
print(f"   Your Mean CV Score: {student_cv_mean:.4f}")

# Visualize the variation between folds
plot_cv_variance(cv_results, LUCKY_SCORE)

# Grading Task 2
if abs(student_cv_mean - TARGET_CV_MEAN) < 0.001:
    print("   ✅ Task 2: PASS")
    score2 = 1
else:
    print(f"   ❌ Task 2: FAIL (Expected mean ~{TARGET_CV_MEAN:.4f})")
    score2 = 0

# ==========================================
# TASK 3: MEASURING STABILITY
# ==========================================
print("\n--- TASK 3: Calculating Model Stability ---")
print("Instructions: Calculate the Standard Deviation of your CV scores.")

# [TODO] TASK 3: Use np.std() on your cv_results
student_std = 0.0  # <--- CHANGE THIS (Hint: np.std(cv_results))

# Grading Task 3
if abs(student_std - np.std(TARGET_CV_SCORES)) < 0.0001:
    print(f"   ✅ Task 3: PASS (Standard Deviation = {student_std:.4f})")
    score3 = 1
else:
    print(f"   ❌ Task 3: FAIL (Check your standard deviation calculation)")
    score3 = 0

# ==========================================
# TASK 4: CONCEPTUAL QUIZ
# ==========================================
print("\n--- TASK 4: FUNDAMENTAL QUIZ ---")
print("Instructions: Replace 'None' with True or False for each statement.")

# [TODO] TASK 4: Quiz
student_quiz_answers = [
    None, # Q1: CV is used to ensure we don't rely on a single 'lucky' split.
    None, # Q2: K-Fold CV is faster to calculate than a single hold-out split.
    None, # Q3: CV helps detect if a model is overfitting to a specific subset of data.
    None, # Q4: Using K=100 is always better than K=5, regardless of dataset size.
    None  # Q5: A low standard deviation across folds indicates a stable model.
]

# Grading Quiz
quiz_points = 0
if None not in student_quiz_answers:
    for i, (ans, correct) in enumerate(zip(student_quiz_answers, QUIZ_KEY)):
        if ans == correct:
            quiz_points += 1
        else:
            print(f"   🚩 Question {i+1} is incorrect.")
else:
    print("   ⚠️ Quiz incomplete. Please answer all questions.")

# ==========================================
# FINAL RESULTS
# ==========================================
total_coding = score1 + score2 + score3
print("\n" + "="*40)
print(f"FINAL SCORECARD")
print(f"Coding Tasks: {total_coding}/3")
print(f"Quiz Score:   {quiz_points}/5")
print("="*40)

if total_coding == 3 and quiz_points == 5:
    print("🎉 PERFECT SCORE! You have mastered the fundamentals of Cross-Validation.")
else:
    print("Keep tweaking! Review the failed tasks or incorrect quiz answers.")