import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
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
    with open('solutions_gmm.pkl', 'rb') as f:
        pkg = pickle.load(f)
        X = pkg['data']
        TARGET_LB = pkg['converged_score']
        TARGET_MEANS = pkg['means']
        QUIZ_KEY = pkg['quiz_key']
    print("✅ GMM Lab Data Loaded.")
except FileNotFoundError:
    print("❌ ERROR: 'solutions_gmm.pkl' not found.")
    exit()

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance handleing all GMM types"""
    ax = ax or plt.gca()
    
    # Handle 'full' covariance (2x2 matrix)
    if covariance.ndim == 2:
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    # Handle 'diag' covariance (1D array)
    elif covariance.ndim == 1:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    # Handle 'spherical' covariance (scalar)
    else:
        angle = 0
        width = height = 2 * np.sqrt(covariance)

    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle=angle, **kwargs))

# ==========================================
# TASK 1: THE EM ITERATION PROCESS
# ==========================================
print("\n--- TASK 1: Convergence and Max Iterations ---")


# [TODO] TASK 1: Change max_iter to 100 to reach convergence.
gmm_task1 = GaussianMixture(
    n_components=3, 
    max_iter=1,      # <--- STARTING VALUE (Change to 100)
    random_state=42
)
gmm_task1.fit(X)

print(f"Converged: {gmm_task1.converged_}")
print(f"Log-Likelihood: {gmm_task1.lower_bound_:.4f}")

score1 = 1 if gmm_task1.converged_ else 0

# ==========================================
# TASK 2: COVARIANCE SHAPES
# ==========================================
print("\n--- TASK 2: Covariance Shapes ---")


# [TODO] TASK 2: Change 'covariance_type' from 'spherical' to 'full'.
gmm_task2 = GaussianMixture(
    n_components=3, 
    covariance_type='spherical', # <--- STARTING VALUE (Change to 'full')
    max_iter=100,
    random_state=42
)
gmm_task2.fit(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=gmm_task2.predict(X), s=20, cmap='viridis', zorder=2)
for i in range(gmm_task2.n_components):
    # Pass the correct covariance based on the type
    if gmm_task2.covariance_type == 'full':
        cov = gmm_task2.covariances_[i]
    else:
        cov = gmm_task2.covariances_[i]
    draw_ellipse(gmm_task2.means_[i], cov, alpha=0.2, color='red')
plt.title(f"GMM with {gmm_task2.covariance_type} covariance")
plt.show()

score2 = 1 if gmm_task2.covariance_type == 'full' else 0

# ==========================================
# TASK 3: SOFT ASSIGNMENTS
# ==========================================
print("\n--- TASK 3: Soft Assignment ---")

# [TODO] Change index from 0 to 9 to inspect point #9
sample_index = 0 
probs = gmm_task2.predict_proba(X[sample_index:sample_index+1]) 

print(f"Probability of point #{sample_index} belonging to clusters: \n{probs}")
score3 = 1 if sample_index == 9 else 0

# ==========================================
# TASK 4: THE BIC CRITERION
# ==========================================
print("\n--- TASK 4: The BIC Score ---")

n_components = np.arange(1, 6)
# Initialized with placeholder zeros so it runs immediately
bics = [0, 0, 0, 0, 0] 

# [TODO] Uncomment and complete the list comprehension below:
# bics = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X).bic(X) for n in n_components]

plt.plot(n_components, bics, marker='o')
plt.title('BIC Score (Lower is better)')
plt.show()

score4 = 1 if bics[2] != 0 and bics[2] == min(bics) else 0

# ==========================================
# TASK 5: CONCEPTUAL QUIZ
# ==========================================
print("\n--- TASK 5: GMM & EM QUIZ ---")

# [TODO] Replace None with True or False
student_quiz = [
    None, # Q1: The 'E' in EM stands for 'Estimation'.
    None, # Q2: GMM can identify overlapping clusters better than K-Means.
    None, # Q3: K-Means is a special case of GMM with spherical, equal-variance clusters.
    None, # Q4: The EM algorithm is guaranteed to find the global optimum every time.
    None  # Q5: 'Soft assignment' means a point can belong to multiple clusters with different probabilities.
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