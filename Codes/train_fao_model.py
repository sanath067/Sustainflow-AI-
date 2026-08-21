import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import joblib


DATA_PATH = os.path.join(
    "Datasets",
    "Fao_merged_clean.csv"
)

RESULTS_DIR = "Results"

RANDOM_STATE = 42


CATEGORICAL_FEATURES = [
    "commodity",
    "country",
    "food_supply_stage"
]

NUMERIC_FEATURES = [
    "year"
]

TARGET = "loss_percentage"


def load_data(path=DATA_PATH):

    df = pd.read_csv(path)

    X = df[
        CATEGORICAL_FEATURES +
        NUMERIC_FEATURES
    ]

    y = df[TARGET]

    return X, y, df


def build_pipeline(model):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES
            )
        ],
        remainder="passthrough"
    )

    return Pipeline(
        steps=[
            (
                "preprocess",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )


def evaluate(
    name,
    y_test,
    y_pred
):

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    print(
        f"\n--- {name}: held-out test set ---"
    )

    print(
        f"RMSE: {rmse:.3f}"
    )

    print(
        f"MAE:  {mae:.3f}"
    )

    print(
        f"R^2:  {r2:.3f}"
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def per_stage_r2(
    X_test,
    y_test,
    y_pred
):

    out = X_test.copy()

    out["actual"] = y_test.values

    out["predicted"] = y_pred

    lines = []

    for stage, group in out.groupby(
        "food_supply_stage"
    ):

        if len(group) < 3:

            lines.append(
                f"  {stage}: n={len(group)} "
                f"(too few test rows for a stable R^2)"
            )

            continue

        r2 = r2_score(
            group["actual"],
            group["predicted"]
        )

        lines.append(
            f"  {stage}: n={len(group)}  "
            f"R^2={r2:.3f}"
        )

    return lines


def save_plots(
    y_test,
    y_pred,
    pipeline,
    results_dir=RESULTS_DIR
):

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    plt.figure(
        figsize=(6, 6)
    )

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.3,
        s=10
    )

    lims = [
        0,
        100
    ]

    plt.plot(
        lims,
        lims,
        "r--",
        label="Perfect prediction"
    )

    plt.xlabel(
        "Actual loss_percentage"
    )

    plt.ylabel(
        "Predicted loss_percentage"
    )

    plt.title(
        "FAO Food Loss (merged data): "
        "Actual vs Predicted (Random Forest)"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            results_dir,
            "actual_vs_predicted.png"
        ),
        dpi=150
    )

    plt.close()

    ohe = (
        pipeline
        .named_steps["preprocess"]
        .named_transformers_["cat"]
    )
    cat_feature_names = list(
        ohe.get_feature_names_out(
            CATEGORICAL_FEATURES
        )
    )

    all_feature_names = (
        cat_feature_names +
        NUMERIC_FEATURES
    )

    importances = (
        pipeline
        .named_steps["model"]
        .feature_importances_
    )

    imp_df = pd.DataFrame(
        {
            "feature": all_feature_names,
            "importance": importances
        }
    )

    imp_df = (
        imp_df
        .sort_values(
            "importance",
            ascending=False
        )
        .head(15)
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.barh(
        imp_df["feature"][::-1],
        imp_df["importance"][::-1]
    )

    plt.xlabel(
        "Importance"
    )

    plt.title(
        "Top 15 Feature Importances "
        "(Random Forest, merged data)"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            results_dir,
            "feature_importance.png"
        ),
        dpi=150
    )

    plt.close()

    print(
        f"Saved plots to: {results_dir}/"
    )


def main():

    X, y, df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    ridge_pipeline = build_pipeline(
        Ridge(alpha=1.0)
    )

    ridge_pipeline.fit(
        X_train,
        y_train
    )

    ridge_pred = ridge_pipeline.predict(
        X_test
    )

    ridge_metrics = evaluate(
        "Ridge (baseline)",
        y_test,
        ridge_pred
    )

    rf_pipeline = build_pipeline(
        RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )

    rf_pipeline.fit(
        X_train,
        y_train
    )

    rf_pred = rf_pipeline.predict(
        X_test
    )

    rf_metrics = evaluate(
        "Random Forest",
        y_test,
        rf_pred
    )

    stage_lines = per_stage_r2(
        X_test,
        y_test,
        rf_pred
    )

    print(
        "\n--- Random Forest R^2 by food_supply_stage ---"
    )

    for line in stage_lines:
        print(line)

    save_plots(
        y_test,
        rf_pred,
        rf_pipeline
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    joblib.dump(
        rf_pipeline,
        os.path.join(
            RESULTS_DIR,
            "fao_model.joblib"
        )
    )

    with open(
        os.path.join(
            RESULTS_DIR,
            "metrics.txt"
        ),
        "w"
    ) as f:

        f.write(
            "FAO Food Loss Prediction - "
            "Model Evaluation (merged real data)\n"
        )

        f.write(
            "=" * 60 + "\n"
        )

        f.write(
            "Split method: plain train_test_split "
            "(no synthetic data, no leakage risk)\n"
        )

        f.write(
            "Features: commodity (raw), country (raw), "
            "food_supply_stage (raw), year\n"
        )

        f.write(
            f"Rows used (real only): {len(df)}\n"
        )

        f.write(
            f"Unique commodities: "
            f"{df['commodity'].nunique()}\n"
        )

        f.write(
            f"Unique countries: "
            f"{df['country'].nunique()}\n\n"
        )

        f.write(
            "Ridge (linear baseline):\n"
        )

        f.write(
            f"  RMSE: "
            f"{ridge_metrics['rmse']:.3f}  "
            f"MAE: "
            f"{ridge_metrics['mae']:.3f}  "
            f"R^2: "
            f"{ridge_metrics['r2']:.3f}\n\n"
        )

        f.write(
            "Random Forest:\n"
        )

        f.write(
            f"  RMSE: "
            f"{rf_metrics['rmse']:.3f}  "
            f"MAE: "
            f"{rf_metrics['mae']:.3f}  "
            f"R^2: "
            f"{rf_metrics['r2']:.3f}\n\n"
        )

        f.write(
            "Random Forest R^2 by food_supply_stage:\n"
        )

        for line in stage_lines:
            f.write(line + "\n")

        f.write(
            "\nSample predictions "
            "(first 10 test rows, Random Forest):\n"
        )

        for i in range(
            min(10, len(y_test))
        ):

            row = X_test.iloc[i]

            f.write(
                f"  Commodity: "
                f"{row['commodity']} | "
                f"Country: "
                f"{row['country']} | "
                f"Stage: "
                f"{row['food_supply_stage']} | "
                f"Year: "
                f"{row['year']} | "
                f"Actual: "
                f"{y_test.iloc[i]:.2f}% | "
                f"Predicted: "
                f"{rf_pred[i]:.2f}%\n"
            )

    print(
        f"\nModel saved to: "
        f"{RESULTS_DIR}/fao_model.joblib"
    )

    print(
        f"Metrics saved to: "
        f"{RESULTS_DIR}/metrics.txt"
    )


if __name__ == "__main__":
    main()