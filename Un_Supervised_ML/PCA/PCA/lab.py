import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
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
# LAB SETUP
# ==========================================
try:
    with open('solutions_pca.pkl', 'rb') as f:
        pkg = pickle.load(f)
        X = pkg['data']
        TARGET_EVR = pkg['explained_variance']
        QUIZ_KEY = pkg['quiz_key']
    print("✅ PCA Lab Data Loaded.")
except FileNotFoundError:
    print("❌ ERROR: 'solutions_pca.pkl' not found.")
    exit()

# ==========================================
# TASK 1: VISUALIZING 3D DATA
# ==========================================
print("\n--- TASK 1: Observe / Visualize High-Dimensional Data ---")
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.6)
ax.set_title("Original 3D Data")
plt.show()

print("Notice that the data looks like a flat pancake in 3D space.")
score1 = 1 # It is just an Observation task

# ==========================================
# TASK 2: REDUCING DIMENSIONS
# ==========================================
print("\n--- TASK 2: Dimensionality Reduction ---")


# [TODO] Change n_components from 3 to 2 to project the data into 2D.
pca_task2 = PCA(n_components=3) # <--- STARTING VALUE (Change to 2)
X_pca = pca_task2.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, c='green')
plt.title(f"Data Reduced to {pca_task2.n_components} Components")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

score2 = 1 if pca_task2.n_components == 2 else 0

# ==========================================
# TASK 3: EXPLAINED VARIANCE (SCREE PLOT)
# ==========================================
print("\n--- TASK 3: Explained Variance Ratio ---")


# [TODO] Use the attribute 'explained_variance_ratio_' from your pca_task2 model.
# This tells you how much 'information' each component holds.
evr = np.array([0, 0, 0]) # <--- STARTING VALUE (Replace with pca_task2.explained_variance_ratio_)

plt.bar(range(1, len(evr)+1), evr)
plt.xlabel('Principal Component')
plt.ylabel('Variance Ratio')
plt.title('Scree Plot')
plt.show()

# Check if the first component captures > 70% of variance
score3 = 1 if evr[0] > 0.7 else 0

# ==========================================
# TASK 4: DATA RECONSTRUCTION (INVERSE TRANSFORM)
# ==========================================
print("\n--- TASK 4: Compression and Loss ---")
# PCA is a form of compression. We lose some detail when we reduce dimensions.
# [TODO] Use pca_task2.inverse_transform(X_pca) to bring data back to 3D space.
X_reconstructed = X # <--- STARTING VALUE (Replace with inverse_transform)

recon_error = np.mean(np.square(X - X_reconstructed))
print(f"Reconstruction Error (MSE): {recon_error:.6f}")

# Score passes if they actually performed the transform (error will be small but > 0)
score4 = 1 if 0 < recon_error < 0.1 else 0

# ==========================================
# TASK 5: PCA QUIZ
# ==========================================
print("\n--- TASK 5: PCA FUNDAMENTALS QUIZ ---")

# [TODO] Replace None with True or False
student_quiz = [
    None, # Q1: PCA is a supervised learning algorithm.
    None, # Q2: The first principal component captures the maximum variance in the data.
    None, # Q3: PCA requires data scaling (mean centering) to work correctly.
    None, # Q4: Principal components are always orthogonal (perpendicular) to each other.
    None  # Q5: Reducing dimensions with PCA always increases model accuracy.
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
print(f"LAB SCORE: {score1 + score2 + score3 + score4}/4 Coding Tasks")
print(f"QUIZ SCORE: {quiz_points}/5")
print("="*40)