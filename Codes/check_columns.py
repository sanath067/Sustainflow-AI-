import pandas as pd

raw = pd.read_csv("Datasets/Fao_raw.csv")
extra = pd.read_csv("Datasets/Fao_raw_extra.csv")

print("Fao_raw.csv columns:")
print(list(raw.columns))
print(f"Rows: {len(raw)}\n")

print("Fao_raw_extra.csv columns:")
print(list(extra.columns))
print(f"Rows: {len(extra)}\n")

print("Common columns:", set(raw.columns) & set(extra.columns))
print("Only in Fao_raw.csv:", set(raw.columns) - set(extra.columns))
print("Only in Fao_raw_extra.csv:", set(extra.columns) - set(raw.columns))

print(raw[["loss_percentage", "loss_percentage.1", "loss_percentage.2", "loss_percentage.3"]].head(10))