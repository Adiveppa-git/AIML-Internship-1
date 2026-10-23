
import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# Scalability graph
df = pd.read_csv("results/scalability_results.csv")

plt.figure(figsize=(9, 5))

for library in ["Pandas", "Polars"]:
    subset = df[df["Library"] == library]
    plt.plot(
        subset["Requested_Rows"],
        subset["Execution_Time_Seconds"],
        marker="o",
        linewidth=2,
        label=library,
    )

plt.title("Pandas vs Polars: Scalability")
plt.xlabel("Number of Rows")
plt.ylabel("Execution Time (seconds)")
plt.xticks([50000, 100000, 250000, 500000],
           ["50K", "100K", "250K", "500K"])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/scalability_comparison.png", dpi=200)
plt.close()

# Peak process memory graph
memory = pd.read_csv("results/peak_memory_results.csv")
pivot = memory.pivot(
    index="Operation",
    columns="Library",
    values="Median_Peak_Process_RSS_MB",
)

ax = pivot[["pandas", "polars"]].plot(
    kind="bar",
    figsize=(9, 5),
)

ax.set_title("Pandas vs Polars: Peak Process Memory")
ax.set_xlabel("Operation")
ax.set_ylabel("Median Peak Process Memory (MB)")
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("results/peak_memory_comparison.png", dpi=200)
plt.close()

print("Graphs saved successfully:")
print("results/scalability_comparison.png")
print("results/peak_memory_comparison.png")
