import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

out = Path("results/xgboost")
out.mkdir(parents=True, exist_ok=True)

# Verified XGBoost test-set confusion matrix
cm = np.array([
    [575, 1],
    [3, 506]
])

# 1. Confusion Matrix
fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(cm)

ax.set_title("XGBoost Confusion Matrix", fontsize=16)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

ax.set_xticks([0, 1])
ax.set_xticklabels(["Normal", "Cheating"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["Normal", "Cheating"])

for i in range(2):
    for j in range(2):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=18
        )

fig.tight_layout()
fig.savefig(
    out / "xgboost_confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)

# 2. Performance Metrics
metrics = {
    "Accuracy": 99.63,
    "Precision": 99.80,
    "Recall": 99.41,
    "F1 Score": 99.61,
    "ROC-AUC": 99.95
}

fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(
    list(metrics.keys()),
    list(metrics.values())
)

ax.set_title("XGBoost Test Performance", fontsize=16)
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 105)

for bar, value in zip(bars, metrics.values()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=11
    )

fig.tight_layout()
fig.savefig(
    out / "xgboost_metrics.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)

# 3. Prediction Summary
labels = [
    "Normal\nCorrect",
    "Normal ->\nCheating",
    "Cheating ->\nNormal",
    "Cheating\nCorrect"
]

values = [575, 1, 3, 506]

fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(labels, values)

ax.set_title(
    "XGBoost Test Prediction Summary",
    fontsize=16
)
ax.set_ylabel("Test Samples")

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 8,
        str(value),
        ha="center",
        va="bottom",
        fontsize=11
    )

fig.tight_layout()
fig.savefig(
    out / "xgboost_test_prediction_summary.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)

print("XGBoost result images created successfully.")
