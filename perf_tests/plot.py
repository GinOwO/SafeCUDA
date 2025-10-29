import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1) Load
# cols: test,description,compiler,run,time_ms
df = pd.read_csv("perf_tests/results/perf_results_2025-10-30_01-49-36.csv")

# 2) Choose percentiles
LOW_Q, HIGH_Q = 0.05, 0.95  # change to 0.01/0.99 if you want 1st/99th

# 3) Aggregate per (test, compiler)
agg = (
    df.groupby(["test", "description", "compiler"])["time_ms"]
    .agg(
        mean="mean",
        median="median",
        p_low=lambda s: s.quantile(LOW_Q),
        p_high=lambda s: s.quantile(HIGH_Q),
        tmin="min",
        tmax="max",
        n="count",
    )
    .reset_index()
)

# 4) Pivot to wide for overhead calc (using mean)
wide_mean = agg.pivot(
    index=["test", "description"], columns="compiler", values="mean"
).reset_index()

# 5) Compute overhead % (mean-based)
if {"nvcc", "sfnvcc"}.issubset(wide_mean.columns):
    wide_mean["overhead_pct"] = (
        (wide_mean["sfnvcc"] - wide_mean["nvcc"]) / wide_mean["nvcc"] * 100.0
    )
else:
    raise ValueError("Expected compilers 'nvcc' and 'sfnvcc' not found")

# 6) For error bars, use the percentile spans per compiler
p_low = agg.pivot(
    index=["test", "description"], columns="compiler", values="p_low"
).reset_index()
p_high = agg.pivot(
    index=["test", "description"], columns="compiler", values="p_high"
).reset_index()

# 7) Merge for plotting convenience
plot_df = (
    wide_mean.merge(
        agg.pivot(
            index=["test", "description"], columns="compiler", values="median"
        ).reset_index(),
        on=["test", "description"],
        suffixes=("", "_median"),
    )
    .merge(p_low, on=["test", "description"], suffixes=("", "_plow"))
    .merge(p_high, on=["test", "description"], suffixes=("", "_phigh"))
)

# 8) Figure A: Grouped bars with error bars and overhead line (dual axis)
tests = plot_df["test"]
x = np.arange(len(tests))
width = 0.35

fig, ax1 = plt.subplots(figsize=(12, 6))
# Bars
nvcc_means = plot_df["nvcc"].values
sfnvcc_means = plot_df["sfnvcc"].values

# Error bars from percentiles: low/high -> asymmetric
nvcc_yerr = np.vstack(
    [
        nvcc_means - plot_df["nvcc_plow"].values,
        plot_df["nvcc_phigh"].values - nvcc_means,
    ]
)
sfnvcc_yerr = np.vstack(
    [
        sfnvcc_means - plot_df["sfnvcc_plow"].values,
        plot_df["sfnvcc_phigh"].values - sfnvcc_means,
    ]
)

b1 = ax1.bar(
    x - width / 2, nvcc_means, width, yerr=nvcc_yerr, capsize=3, label="nvcc (mean)"
)
b2 = ax1.bar(
    x + width / 2,
    sfnvcc_means,
    width,
    yerr=sfnvcc_yerr,
    capsize=3,
    label="sfnvcc (mean)",
)
ax1.set_ylabel("Time (ms)")
ax1.set_xticks(x)
ax1.set_xticklabels(tests, rotation=0)
ax1.legend(loc="upper left")

# Overhead line on second axis
ax2 = ax1.twinx()
ax2.plot(
    x,
    plot_df["overhead_pct"].values,
    color="crimson",
    marker="o",
    linewidth=2,
    label="Overhead (%)",
)
ax2.set_ylabel("Overhead (%)")
ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax2.legend(loc="upper right")
fig.tight_layout()
fig.savefig("perf_tests/output/perf_bars_overhead.png", dpi=200)

# 9) Figure B: Boxplots per test per compiler (medians visible)
fig2, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(
    data=df, x="test", y="time_ms", hue="compiler", whis=[5, 95], ax=ax
)  # whiskers at p5/p95
ax.set_ylabel("Time (ms)")
ax.set_xlabel("Test")
ax.set_title("Per-test distributions (medians & percentile whiskers)")
fig2.tight_layout()
fig2.savefig("perf_tests/output/perf_boxplots.png", dpi=200)

# 10) Optional: summary CSV
summary = agg.merge(
    wide_mean[["test", "description", "overhead_pct"]],
    on=["test", "description"],
    how="left",
)
summary.to_csv("perf_tests/output/perf_summary_stats.csv", index=False)
