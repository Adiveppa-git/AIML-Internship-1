
import os
import time
import statistics
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
CSV_FILE = "Online Retail.csv"
REPEATS = 5
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Dataset not found: {CSV_FILE}")

print("=" * 60)
print("POLARS VS PANDAS BENCHMARK")
print("=" * 60)

# --------------------------------------------------
# 2. Dataset loading functions
# --------------------------------------------------
def load_pandas():
    return pd.read_csv(
        CSV_FILE,
        dtype={
            "InvoiceNo": "string",
            "StockCode": "string",
            "Description": "string",
            "Country": "string",
        },
    )


def load_polars():
    return pl.read_csv(
        CSV_FILE,
        infer_schema_length=10000,
        schema_overrides={
            "InvoiceNo": pl.String,
            "StockCode": pl.String,
            "Description": pl.String,
            "Country": pl.String,
        },
    )


# --------------------------------------------------
# 3. Load and inspect the dataset
# --------------------------------------------------
df_pd = load_pandas()
df_pl = load_polars()

print("\nDATASET INFORMATION")
print("File size (MB):", round(os.path.getsize(CSV_FILE) / (1024 ** 2), 2))
print("Pandas shape:", df_pd.shape)
print("Polars shape:", df_pl.shape)

assert df_pd.shape == (541909, 8)
assert df_pl.shape == df_pd.shape

# --------------------------------------------------
# 4. Prepare equivalent lookup tables for joining
# --------------------------------------------------
products_pd = (
    df_pd[["StockCode", "Description"]]
    .dropna(subset=["StockCode"])
    .drop_duplicates(subset=["StockCode"])
)

products_pl = (
    df_pl.select(["StockCode", "Description"])
    .filter(pl.col("StockCode").is_not_null())
    .unique(subset=["StockCode"], keep="first")
)

# --------------------------------------------------
# 5. Define equivalent operations
# --------------------------------------------------
def pd_filter(df):
    return df.loc[
        (df["Quantity"] > 0) & (df["UnitPrice"] > 0)
    ]


def pl_filter(df):
    return df.filter(
        (pl.col("Quantity") > 0) & (pl.col("UnitPrice") > 0)
    )


def pd_select(df):
    return df[[
        "InvoiceNo", "StockCode", "Quantity",
        "UnitPrice", "Country"
    ]]


def pl_select(df):
    return df.select([
        "InvoiceNo", "StockCode", "Quantity",
        "UnitPrice", "Country"
    ])


def pd_group(df):
    temp = df.assign(Revenue=df["Quantity"] * df["UnitPrice"])
    return (
        temp.groupby("Country", dropna=False)
        .agg(
            TotalQuantity=("Quantity", "sum"),
            TotalRevenue=("Revenue", "sum"),
        )
        .reset_index()
    )


def pl_group(df):
    return (
        df.with_columns(
            (pl.col("Quantity") * pl.col("UnitPrice")).alias("Revenue")
        )
        .group_by("Country")
        .agg(
            pl.col("Quantity").sum().alias("TotalQuantity"),
            pl.col("Revenue").sum().alias("TotalRevenue"),
        )
    )


def pd_join(df):
    return df.merge(
        products_pd,
        on="StockCode",
        how="left",
        suffixes=("", "_product"),
    )


def pl_join(df):
    return df.join(
        products_pl,
        on="StockCode",
        how="left",
        suffix="_product",
    )


def pd_sort_transform(df):
    return (
        df.assign(Revenue=df["Quantity"] * df["UnitPrice"])
        .sort_values("Revenue", ascending=False)
        .reset_index(drop=True)
    )


def pl_sort_transform(df):
    return (
        df.with_columns(
            (pl.col("Quantity") * pl.col("UnitPrice")).alias("Revenue")
        )
        .sort("Revenue", descending=True)
    )


# --------------------------------------------------
# 6. Benchmark helper
# --------------------------------------------------
def benchmark(function, *args):
    # One warm-up run, excluded from timing
    function(*args)

    times = []
    result = None

    for _ in range(REPEATS):
        start = time.perf_counter()
        result = function(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return statistics.median(times), result


def dataframe_memory_mb(df, library):
    # Approximate memory occupied by the resulting DataFrame.
    if library == "Pandas":
        return df.memory_usage(index=True, deep=True).sum() / (1024 ** 2)

    return df.estimated_size(unit="mb")


# --------------------------------------------------
# 7. Benchmark all required operations
# --------------------------------------------------
operations = [
    ("Load CSV", load_pandas, load_polars, False),
    ("Filter rows", pd_filter, pl_filter, True),
    ("Select columns", pd_select, pl_select, True),
    ("Group and aggregate", pd_group, pl_group, True),
    ("Join datasets", pd_join, pl_join, True),
    ("Sort and transform", pd_sort_transform, pl_sort_transform, True),
]

records = []

print("\nBENCHMARK RESULTS")
print("Each operation is repeated", REPEATS, "times.")
print("Times below are medians; warm-up runs are excluded.\n")

for name, pd_func, pl_func, uses_data in operations:
    if uses_data:
        pd_time, pd_result = benchmark(pd_func, df_pd)
        pl_time, pl_result = benchmark(pl_func, df_pl)
    else:
        pd_time, pd_result = benchmark(pd_func)
        pl_time, pl_result = benchmark(pl_func)

    pd_memory = dataframe_memory_mb(pd_result, "Pandas")
    pl_memory = dataframe_memory_mb(pl_result, "Polars")

    # Basic output checks
    if name == "Filter rows":
        assert len(pd_result) == pd_result.shape[0]
        assert len(pd_result) == pl_result.height

    if name == "Group and aggregate":
        assert len(pd_result) == pl_result.height

    if name == "Join datasets":
        assert len(pd_result) == pl_result.height

    records.append({
        "Operation": name,
        "Pandas_Median_Seconds": pd_time,
        "Polars_Median_Seconds": pl_time,
        "Pandas_Output_Memory_MB": pd_memory,
        "Polars_Output_Memory_MB": pl_memory,
        "Pandas_Output_Rows": len(pd_result),
        "Polars_Output_Rows": pd_result.height
        if isinstance(pd_result, pl.DataFrame)
        else pl_result.height,
    })

    print(f"{name}:")
    print(f"  Pandas median: {pd_time:.6f} seconds")
    print(f"  Polars median: {pl_time:.6f} seconds")
    print(f"  Pandas output memory: {pd_memory:.2f} MB")
    print(f"  Polars output memory: {pl_memory:.2f} MB")
    print()

# --------------------------------------------------
# 8. Save benchmark results
# --------------------------------------------------
results = pd.DataFrame(records)
results.to_csv(f"{RESULTS_DIR}/benchmark_results.csv", index=False)

print("RESULTS TABLE")
print(results.round(4).to_string(index=False))

# --------------------------------------------------
# 9. Plot execution-time comparison
# --------------------------------------------------
ax = results.set_index("Operation")[
    ["Pandas_Median_Seconds", "Polars_Median_Seconds"]
].plot(kind="bar", figsize=(12, 6))

ax.set_title("Pandas vs Polars: Median Execution Time")
ax.set_ylabel("Seconds (lower is better)")
ax.set_xlabel("Operation")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/execution_time_comparison.png", dpi=150)
plt.close()

# --------------------------------------------------
# 10. Plot output DataFrame memory comparison
# --------------------------------------------------
ax = results.set_index("Operation")[
    ["Pandas_Output_Memory_MB", "Polars_Output_Memory_MB"]
].plot(kind="bar", figsize=(12, 6))

ax.set_title("Pandas vs Polars: Output DataFrame Memory")
ax.set_ylabel("Approximate memory (MB; lower is better)")
ax.set_xlabel("Operation")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/memory_comparison.png", dpi=150)
plt.close()

# --------------------------------------------------
# 11. Summary
# --------------------------------------------------
print("\nFILES CREATED")
print(f"{RESULTS_DIR}/benchmark_results.csv")
print(f"{RESULTS_DIR}/execution_time_comparison.png")
print(f"{RESULTS_DIR}/memory_comparison.png")

print("\nBenchmark completed successfully.")
