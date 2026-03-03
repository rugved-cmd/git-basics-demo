import numpy as np

# ----------------------------
# Task 1: Generate & Inspect
# ----------------------------
np.random.seed(42)

scores = np.random.randint(50, 101, size=(5, 4))

print("Scores:\n", scores)

print("\n3rd student, 2nd subject:", scores[2, 1])

print("\nLast 2 students:\n", scores[-2:, :])

print("\nFirst 3 students, subjects 2 & 3:\n", scores[:3, 1:3])

# ----------------------------
# Task 2: Broadcasting
# ----------------------------

column_means = np.round(scores.mean(axis=0), 2)
print("\nColumn-wise Mean:", column_means)

curve = np.array([5, 3, 7, 2])
curved_scores = scores + curve

curved_scores = np.clip(curved_scores, None, 100)

print("\nCurved Scores:\n", curved_scores)

row_wise_max = curved_scores.max(axis=1)
print("\nRow-wise Max:", row_wise_max)

# ----------------------------
# Task 3: Normalization
# ----------------------------

row_min = curved_scores.min(axis=1, keepdims=True)
row_max = curved_scores.max(axis=1, keepdims=True)

normalized = (curved_scores - row_min) / (row_max - row_min)

print("\nNormalized Array:\n", normalized)

max_index = np.unravel_index(np.argmax(normalized), normalized.shape)
print("\nIndex of Highest Value:", max_index)

above_90 = curved_scores[curved_scores > 90]
print("\nCurved Scores > 90:", above_90)