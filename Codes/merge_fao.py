import os
import pandas as pd

DATASETS_DIR = "Datasets"
RAW_PATH = os.path.join(DATASETS_DIR, "Fao_raw.csv")
EXTRA_PATH = os.path.join(DATASETS_DIR, "Fao_raw_extra.csv")
OUT_RAW_PATH = os.path.join(DATASETS_DIR, "Fao_merged_raw.csv")
OUT_CLEAN_PATH = os.path.join(DATASETS_DIR, "Fao_merged_clean.csv")

KEY_COLUMNS = ["commodity", "country", "food_supply_stage", "year"]
KEEP_COLUMNS = KEY_COLUMNS + ["loss_percentage"]


def load_and_align(path):
    df = pd.read_csv(path)
    df = df[KEEP_COLUMNS].copy()
    for col in ["commodity", "country", "food_supply_stage"]:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def main():
    raw_df = load_and_align(RAW_PATH)
    extra_df = load_and_align(EXTRA_PATH)

    print(f"Fao_raw.csv rows: {len(raw_df)}")
    print(f"Fao_raw_extra.csv rows: {len(extra_df)}")

    merged_raw = pd.concat([raw_df, extra_df], ignore_index=True, sort=False)
    print(f"Combined (before dedup): {len(merged_raw)}")

    before = len(merged_raw)
    merged_raw = merged_raw.drop_duplicates(
        subset=KEEP_COLUMNS, keep="first"
    )
    print(f"Dropped {before - len(merged_raw)} exact duplicate rows")

    os.makedirs(DATASETS_DIR, exist_ok=True)
    merged_raw.to_csv(OUT_RAW_PATH, index=False)
    print(f"Saved combined raw data to: {OUT_RAW_PATH}")

    merged_clean = merged_raw.dropna(subset=KEEP_COLUMNS).copy()
    merged_clean = merged_clean[merged_clean["loss_percentage"].between(0, 100)]
    merged_clean.to_csv(OUT_CLEAN_PATH, index=False)
    print(f"Saved cleaned merged data to: {OUT_CLEAN_PATH}")
    print(f"Final clean row count: {len(merged_clean)}")


if __name__ == "__main__":
    main()

