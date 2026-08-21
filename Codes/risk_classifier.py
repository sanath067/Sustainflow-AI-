import pandas as pd


def classify_risk(loss_percentage):
    """
    Classify a predicted or actual loss_percentage into Low/Medium/High risk.
    Thresholds derived from the merged FAO dataset distribution:
      - Low:    < 5%   (~75th percentile of real data)
      - Medium: 5-15%  (up to just above the 90th percentile)
      - High:   > 15%  (long-tail severe loss cases)
    """
    if loss_percentage < 5:
        return "Low"
    elif loss_percentage < 15:
        return "Medium"
    else:
        return "High"


def add_risk_column(df, loss_col="predicted_loss_percentage"):
    df = df.copy()
    df["risk_level"] = df[loss_col].apply(classify_risk)
    return df


if __name__ == "__main__":
    # quick sanity check against the merged dataset
    df = pd.read_csv("Datasets/Fao_merged_clean.csv")
    df = add_risk_column(df, loss_col="loss_percentage")
    print(df["risk_level"].value_counts())
    print(df["risk_level"].value_counts(normalize=True) * 100)