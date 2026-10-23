
import os
import time
import pandas as pd
import polars as pl

CSV_FILE = "Online Retail.csv"
SIZES = [50000, 100000, 250000, 500000]

def benchmark_pandas(path, nrows):
    start = time.perf_counter()

    df = pd.read_csv(
        path,
        nrows=nrows,
        dtype={
            "InvoiceNo": "string",
            "StockCode": "string",
            "Description": "string",
            "Country": "string",
        },
    )

    result = df.loc[
        (df["Quantity"] > 0) & (df["UnitPrice"] > 0)
    ].copy()

    result["Revenue"] = result["Quantity"] * result["UnitPrice"]
    summary = result.groupby("Country")["Revenue"].sum()

    elapsed = time.perf_counter() - start
    return elapsed, len(df), len(result), len(summary)

def benchmark_polars(path, nrows):
    start = time.perf_counter()

    df = pl.read_csv(
        path,
        n_rows=nrows,
        infer_schema_length=10000,
        schema_overrides={
            "InvoiceNo": pl.String,
            "StockCode": pl.String,
            "Description": pl.String,
            "Country": pl.String,
        },
    )

    result = df.filter(
        (pl.col("Quantity") > 0) & (pl.col("UnitPrice") > 0)
    ).with_columns(
        (pl.col("Quantity") * pl.col("UnitPrice")).alias("Revenue")
    )

    summary = result.group_by("Country").agg(
        pl.col("Revenue").sum().alias("TotalRevenue")
    )

    elapsed = time.perf_counter() - start
    return elapsed, df.height, result.height, summary.height

if __name__ == "__main__":
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(CSV_FILE)

    records = []

    print("Scalability benchmark")
    print("Each time includes CSV loading, filtering, and aggregation.\n")

    for size in SIZES:
        for library, function in [
            ("Pandas", benchmark_pandas),
            ("Polars", benchmark_polars),
        ]:
            try:
                seconds, loaded, filtered, groups = function(CSV_FILE, size)

                records.append({
                    "Requested_Rows": size,
                    "Library": library,
                    "Loaded_Rows": loaded,
                    "Filtered_Rows": filtered,
                    "Country_Groups": groups,
                    "Execution_Time_Seconds": round(seconds, 4),
                })

                print(
                    f"{library:6} | requested={size:>6,} "
                    f"| loaded={loaded:>6,} "
                    f"| time={seconds:.4f}s "
                    f"| filtered={filtered:,}"
                )
            except Exception as error:
                print(f"{library} failed at {size:,} rows: {error}")
                raise

    os.makedirs("results", exist_ok=True)
    output = pd.DataFrame(records)
    output.to_csv("results/scalability_results.csv", index=False)

    print("\nRESULTS")
    print(output.to_string(index=False))
    print("\nSaved: results/scalability_results.csv")
