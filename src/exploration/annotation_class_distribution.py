
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("annotation_by_category.csv")

# Set first column as index
df = df.set_index(df.columns[0])

# Select classes
classes = ["pol", "spo", "paly", "din", "alg", "fun"]
annotation_df = df.loc[classes]

annotation_df = annotation_df.apply(pd.to_numeric, errors="coerce")

# Sum across columns
combined_freq = annotation_df.sum(axis=1)

# Sort
combined_freq = combined_freq.sort_values(ascending=False)

# Compute percentages
total = combined_freq.sum()
percentages = (combined_freq / total) * 100

# ----------- Plot -----------

plt.figure(figsize=(9, 5))
ax = combined_freq.plot(kind="bar")

# Clean look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.xlabel("Annotation Class", fontsize=12)
plt.ylabel("Combined Frequency", fontsize=12)
plt.title("Distribution of Annotation Classes", fontsize=14)

plt.xticks(rotation=0)

# Add labels: count + percentage
for i, (count, pct) in enumerate(zip(combined_freq, percentages)):
    ax.text(i, count,
            f"{int(count)}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10)

plt.tight_layout()
plt.show()