
import pandas as pd
import polars as pl

CSV_FILE = "Online Retail.csv"

pd_df = pd.read_csv(
    CSV_FILE,
    dtype={
        "InvoiceNo": "string",
        "StockCode": "string",
        "Description": "string",
        "Country": "string",
    },
)

pl_df = pl.read_csv(
    CSV_FILE,
    infer_schema_length=10000,
    schema_overrides={
        "InvoiceNo": pl.String,
        "StockCode": pl.String,
        "Description": pl.String,
        "Country": pl.String,
    },
)

print("DATASET CHECK")
print("Pandas shape:", pd_df.shape)
print("Polars shape:", pl_df.shape)

# Check filtering equivalence
pd_filtered = pd_df.loc[
    (pd_df["Quantity"] > 0) & (pd_df["UnitPrice"] > 0)
]

pl_filtered = pl_df.filter(
    (pl.col("Quantity") > 0) & (pl.col("UnitPrice") > 0)
)

assert len(pd_filtered) == pl_filtered.height
print("PASS: Filter row counts match")

# Check selected columns
columns = [
    "InvoiceNo", "StockCode", "Quantity",
    "UnitPrice", "Country"
]

pd_selected = pd_df[columns]
pl_selected = pl_df.select(columns)

assert pd_selected.shape == (
    pl_selected.height, pl_selected.width
)
print("PASS: Selected column shapes match")

# Check aggregation results
pd_grouped = (
    pd_df.assign(
        Revenue=pd_df["Quantity"] * pd_df["UnitPrice"]
    )
    .groupby("Country", dropna=False)
    .agg(
        TotalQuantity=("Quantity", "sum"),
        TotalRevenue=("Revenue", "sum"),
    )
    .sort_index()
)

pl_grouped = (
    pl_df.with_columns(
        (pl.col("Quantity") * pl.col("UnitPrice"))
        .alias("Revenue")
    )
    .group_by("Country")
    .agg(
        pl.col("Quantity").sum().alias("TotalQuantity"),
        pl.col("Revenue").sum().alias("TotalRevenue"),
    )
    .sort("Country")
)

pd_grouped = pd_grouped.sort_index(
    key=lambda index: index.fillna("")
)

assert len(pd_grouped) == pl_grouped.height
assert list(pd_grouped["TotalQuantity"]) == (
    pl_grouped["TotalQuantity"].to_list()
)

for a, b in zip(
    pd_grouped["TotalRevenue"],
    pl_grouped["TotalRevenue"].to_list()
):
    assert abs(float(a) - float(b)) < 0.01

print("PASS: Aggregation values match")

print("\nAll implemented validation checks passed.")
