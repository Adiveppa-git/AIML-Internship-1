
import os
import sys
import time
import subprocess
import psutil
import pandas as pd
import polars as pl

CSV_FILE = "Online Retail.csv"
REPEATS = 3

def run_worker(library, operation):
    """Execute one operation in a fresh process."""
    if library == "pandas":
        df = pd.read_csv(
            CSV_FILE,
            dtype={
                "InvoiceNo": "string",
                "StockCode": "string",
                "Description": "string",
                "Country": "string",
            },
        )
    else:
        df = pl.read_csv(
            CSV_FILE,
            infer_schema_length=10000,
            schema_overrides={
                "InvoiceNo": pl.String,
                "StockCode": pl.String,
                "Description": pl.String,
                "Country": pl.String,
            },
        )

    if operation == "filter":
        if library == "pandas":
            result = df.loc[
                (df["Quantity"] > 0) & (df["UnitPrice"] > 0)
            ]
        else:
            result = df.filter(
                (pl.col("Quantity") > 0)
                & (pl.col("UnitPrice") > 0)
            )

    elif operation == "group":
        if library == "pandas":
            result = (
                df.assign(
                    Revenue=df["Quantity"] * df["UnitPrice"]
                )
                .groupby("Country")
                .agg(
                    TotalQuantity=("Quantity", "sum"),
                    TotalRevenue=("Revenue", "sum"),
                )
            )
        else:
            result = (
                df.with_columns(
                    (pl.col("Quantity") * pl.col("UnitPrice"))
                    .alias("Revenue")
                )
                .group_by("Country")
                .agg(
                    pl.col("Quantity").sum().alias("TotalQuantity"),
                    pl.col("Revenue").sum().alias("TotalRevenue"),
                )
            )

    elif operation == "sort":
        if library == "pandas":
            result = df.assign(
                Revenue=df["Quantity"] * df["UnitPrice"]
            ).sort_values("Revenue", ascending=False)
        else:
            result = df.with_columns(
                (pl.col("Quantity") * pl.col("UnitPrice"))
                .alias("Revenue")
            ).sort("Revenue", descending=True)

    else:
        raise ValueError("Unknown operation")

    # Keep the result alive until the process exits.
    print("Completed:", library, operation, len(result))


def measure_peak(library, operation):
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        library,
        operation,
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    proc = psutil.Process(process.pid)
    peak_rss = 0

    while process.poll() is None:
        try:
            # Include the worker and its child processes.
            rss = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            peak_rss = max(peak_rss, rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        time.sleep(0.02)

    stderr = process.stderr.read() if process.stderr else ""
    if process.returncode != 0:
        raise RuntimeError(
            f"{library} {operation} failed:\n{stderr}"
        )

    return peak_rss / (1024 ** 2)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        run_worker(sys.argv[2], sys.argv[3])
        sys.exit(0)

    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(CSV_FILE)

    operations = ["filter", "group", "sort"]
    records = []

    print("Peak RAM benchmark")
    print("Dataset:", CSV_FILE)
    print("Measurements per operation:", REPEATS)
    print("Note: includes dataset loading and processing.\n")

    for operation in operations:
        for library in ["pandas", "polars"]:
            measurements = []

            for run in range(REPEATS):
                peak_mb = measure_peak(library, operation)
                measurements.append(peak_mb)
                print(
                    f"{library:7} | {operation:6} | "
                    f"run {run + 1}: {peak_mb:.2f} MB"
                )

            records.append({
                "Operation": operation,
                "Library": library,
                "Median_Peak_Process_RSS_MB":
                    sorted(measurements)[len(measurements) // 2],
            })

    results = pd.DataFrame(records)
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/peak_memory_results.csv", index=False)

    print("\nSUMMARY")
    print(results.to_string(index=False))
    print("\nSaved: results/peak_memory_results.csv")
