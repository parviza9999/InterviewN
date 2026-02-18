import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
# LAB SETUP (DO NOT EDIT THIS SECTION)
# ==========================================
try:
    with open('kmeans.pkl', 'rb') as f:
        data = pickle.load(f)
        X = data['dataset']
        correct_inertia_t1 = data['task_1_inertia']
        correct_centers_t1 = data['task_1_centers']
        correct_inertia_t2 = data['task_2_inertia']
    print(f"Loaded dataset with {X.shape[0]} samples.")
except FileNotFoundError:
    print("CRITICAL ERROR: 'kmeans.pkl' not found. Please download it to the same directory.")
    exit()

def score_submission(student_val, correct_val, tolerance=0.1, task_name="Task"):
    """Helper to grade answers within a margin of error."""
    diff = abs(student_val - correct_val)
    if diff < tolerance:
        print(f"✅ {task_name}: CORRECT")
        return 1
    else:
        print(f"❌ {task_name}: INCORRECT (Expected ~{correct_val:.2f}, Got {student_val:.2f})")
        return 0

# ==========================================
# VISUALIZATION HELPER
# ==========================================
def plot_clusters(model, title):
    plt.scatter(X[:, 0], X[:, 1], c=model.labels_, s=30, cmap='viridis')
    centers = model.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title(title)
    plt.show()

# ==========================================
# STUDENT SECTION STARTS HERE
# ==========================================

print("\n--- TASK 1: Finding the Optimal K ---")
print("Instructions: The dataset has distinct blobs. Change 'n_clusters' until the score passes.")
print("Hint: Visualize the data or try values between 2 and 6.")

# [TODO] TASK 1: Tweak n_clusters to match the true structure of the data
# CURRENT SETTING: n_clusters is set to 2 (which is likely wrong)
# ACTION: Change n_clusters to the correct number.
kmeans_task1 = KMeans(
    n_clusters=2,           # <--- TWEAK THIS
    init='k-means++',       # Notice the  initialization is K-means++, we will tweak this later
    n_init=10, 
    random_state=42
)
kmeans_task1.fit(X)
print("Visualizing the clusters, X represents centroid")
plot_clusters(kmeans_task1, "Task 1: Clustering")

# Check Task 1 Dont edit this section
score1 = score_submission(kmeans_task1.inertia_, correct_inertia_t1, task_name="Task 1 (Optimal K)")
if score1:
    print("   Great work! You found the correct number of clusters.")
    
# Check Task 1 complete

print("\n--- TASK 2: Initialization Impact ---")
print("Instructions: Sometimes 'random' initialization creates bad clusters.")
print("We want you to replicate a 'bad' run to understand why defaults matter.")

# [TODO] TASK 2: Replicate a specific bad initialization
# CURRENT SETTING: init is 'k-means++' (which is too good!)
# ACTION: Change 'init' to 'random' and set 'n_init' to 1 (force a single bad attempt)
# NOTE: Keep random_state=1 for grading consistency.
kmeans_task2 = KMeans(
    n_clusters=4, 
    init='k-means++',       # <--- TWEAK THIS to 'random'
    n_init=10,              # <--- TWEAK THIS to 1
    random_state=1
)
kmeans_task2.fit(X)

# Check Task 2
score2 = score_submission(kmeans_task2.inertia_, correct_inertia_t2, task_name="Task 2 (Bad Init)")
if score2:
    print("   Correct! You successfully replicated a suboptimal clustering result.")
    print("   Notice how the red X centroids might be stuck between groups?")
    plot_clusters(kmeans_task2, "Task 2: Suboptimal Clustering (Random Init)")

# ==========================================
# FINAL SCORE
# ==========================================
total_score = score1 + score2
print(f"\nFINAL SCORE: {total_score}/2")
if total_score == 2:
    print("🎉 LAB COMPLETE!")
else:
    print("⚠️  Please fix the incorrect tasks above.")