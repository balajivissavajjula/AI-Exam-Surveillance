import matplotlib.pyplot as plt
from pathlib import Path

out = Path("results/gazenet_v2")
out.mkdir(parents=True, exist_ok=True)

# Verified GazeNet V2 Epoch-2 validation results
metrics = {
    "Mean Angular Error": 11.1583,
    "Median Angular Error": 9.3080,
    "Within 5 deg": 18.5597,
    "Within 10 deg": 54.1862,
    "Within 15 deg": 75.6440
}

# 1. Validation performance
fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(
    list(metrics.keys()),
    list(metrics.values())
)

ax.set_title("GazeNet V2 Validation Results - Epoch 2", fontsize=16)
ax.set_ylabel("Value")
ax.tick_params(axis="x", rotation=20)

for bar, value in zip(bars, metrics.values()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

fig.tight_layout()
fig.savefig(
    out / "gazenet_v2_validation_results.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)

# 2. Angular-error thresholds
thresholds = [
    "Within 5 deg",
    "Within 10 deg",
    "Within 15 deg"
]

values = [
    18.5597,
    54.1862,
    75.6440
]

fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(thresholds, values)

ax.set_title(
    "GazeNet V2 Angular Error Threshold Results",
    fontsize=16
)

ax.set_xlabel("Angular Error Threshold")
ax.set_ylabel("Validation Samples (%)")
ax.set_ylim(0, 100)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=11
    )

fig.tight_layout()
fig.savefig(
    out / "gazenet_v2_angular_error_thresholds.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)

print("GazeNet V2 result images created successfully.")
