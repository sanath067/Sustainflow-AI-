import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = os.path.join(
    "Datasets",
    "Fao_merged_clean.csv"
)

MODEL_PATH = os.path.join(
    "Results",
    "fao_model.joblib"
)

RESULTS_DIR = "Results"

RANDOM_STATE = 42


# ============================================================
# 2. FEATURES
# ============================================================

CATEGORICAL_FEATURES = [
    "commodity",
    "country",
    "food_supply_stage"
]

NUMERIC_FEATURES = [
    "year"
]

FEATURES = (
    CATEGORICAL_FEATURES +
    NUMERIC_FEATURES
)

TARGET = "loss_percentage"


# ============================================================
# 3. SETTINGS
# ============================================================

# Small number of rows so SHAP runs faster
SHAP_SAMPLE_SIZE = 10


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("EXPLAINABLE AI - SHAP ANALYSIS")
print("=" * 60)

print("\nLoading FAO dataset...")

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Dataset rows: {len(df)}"
)

X = df[
    FEATURES
]

y = df[
    TARGET
]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Testing rows: {len(X_test)}"
)


# ============================================================
# 6. LOAD TRAINED MODEL
# ============================================================

print(
    "\nLoading trained Random Forest model..."
)

if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )


pipeline = joblib.load(
    MODEL_PATH
)

print(
    "Model loaded successfully."
)


# ============================================================
# 7. GET PREPROCESSOR AND MODEL
# ============================================================

preprocessor = pipeline.named_steps[
    "preprocess"
]

model = pipeline.named_steps[
    "model"
]


# ============================================================
# 8. SELECT SMALL SAMPLE
# ============================================================

sample_size = min(
    SHAP_SAMPLE_SIZE,
    len(X_test)
)

X_sample = X_test.iloc[
    :sample_size
].copy()

y_sample = y_test.iloc[
    :sample_size
].copy()


print(
    f"\nUsing {sample_size} test rows for SHAP analysis."
)

print(
    "This small sample is intentional so the explanation "
    "runs quickly."
)


# ============================================================
# 9. TRANSFORM SAMPLE DATA
# ============================================================

print(
    "\nTransforming sample data..."
)

X_sample_transformed = preprocessor.transform(
    X_sample
)

print(
    f"Encoded features: "
    f"{X_sample_transformed.shape[1]}"
)


# ============================================================
# 10. GET FEATURE NAMES
# ============================================================

feature_names = list(
    preprocessor.get_feature_names_out()
)


# ============================================================
# 11. CONVERT TO DENSE ARRAY
# ============================================================

if hasattr(
    X_sample_transformed,
    "toarray"
):

    X_sample_dense = (
        X_sample_transformed.toarray()
    )

else:

    X_sample_dense = np.asarray(
        X_sample_transformed
    )


# ============================================================
# 12. CREATE SHAP EXPLAINER
# ============================================================

print(
    "\nCreating SHAP TreeExplainer..."
)

explainer = shap.TreeExplainer(
    model
)

print(
    "SHAP explainer created."
)


# ============================================================
# 13. CALCULATE SHAP VALUES
# ============================================================

print(
    "\nCalculating SHAP values..."
)

# check_additivity=False prevents SHAP from stopping
# because of an internal additivity validation error.

shap_values = explainer.shap_values(
    X_sample_dense,
    check_additivity=False
)

print(
    "SHAP values calculated successfully."
)


# ============================================================
# 14. HANDLE SHAP OUTPUT
# ============================================================

if isinstance(
    shap_values,
    list
):

    shap_values_array = np.asarray(
        shap_values[0]
    )

else:

    shap_values_array = np.asarray(
        shap_values
    )


# ============================================================
# 15. CALCULATE GLOBAL FEATURE IMPORTANCE
# ============================================================

print(
    "\nCalculating global feature importance..."
)

mean_abs_shap = np.mean(
    np.abs(
        shap_values_array
    ),
    axis=0
)


importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }
)


importance_df = (
    importance_df
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
)


# ============================================================
# 16. CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 17. SAVE GLOBAL FEATURE IMPORTANCE
# ============================================================

importance_path = os.path.join(
    RESULTS_DIR,
    "shap_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)

print(
    "\nGlobal SHAP importance saved to:"
)

print(
    importance_path
)


# ============================================================
# 18. CREATE GLOBAL IMPORTANCE PLOT
# ============================================================

print(
    "\nCreating global SHAP importance plot..."
)

top_n = min(
    15,
    len(importance_df)
)

plot_df = (
    importance_df
    .head(top_n)
    .sort_values(
        "mean_abs_shap",
        ascending=True
    )
)


plt.figure(
    figsize=(10, 6)
)

plt.barh(
    plot_df["feature"],
    plot_df["mean_abs_shap"]
)

plt.xlabel(
    "Mean Absolute SHAP Value"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top Features Affecting Food Loss Prediction"
)

plt.tight_layout()


global_plot_path = os.path.join(
    RESULTS_DIR,
    "shap_global_importance.png"
)

plt.savefig(
    global_plot_path,
    dpi=150
)

plt.close()


print(
    "Global SHAP plot saved to:"
)

print(
    global_plot_path
)


# ============================================================
# 19. CREATE LOCAL EXPLANATIONS
# ============================================================

print(
    "\nCreating local explanations..."
)

local_rows = []


for row_index in range(
    sample_size
):

    row_shap = (
        shap_values_array[
            row_index
        ]
    )


    # Sort features by importance
    sorted_indices = np.argsort(
        np.abs(
            row_shap
        )
    )[::-1]


    # Keep the top 10 influencing features
    top_indices = sorted_indices[
        :10
    ]


    # Get prediction for this specific row
    prediction = pipeline.predict(
        X_sample.iloc[
            [row_index]
        ]
    )[0]


    actual = y_sample.iloc[
        row_index
    ]


    for rank, feature_index in enumerate(
        top_indices,
        start=1
    ):

        shap_value = row_shap[
            feature_index
        ]


        # Determine whether feature pushes prediction
        # upward or downward

        if shap_value > 0:

            effect = (
                "Increases prediction"
            )

        elif shap_value < 0:

            effect = (
                "Decreases prediction"
            )

        else:

            effect = (
                "No effect"
            )


        local_rows.append(
            {
                "test_row": row_index,
                "rank": rank,
                "feature": feature_names[
                    feature_index
                ],
                "shap_value": shap_value,
                "absolute_shap_value": abs(
                    shap_value
                ),
                "effect": effect,
                "actual_loss_percentage": actual,
                "predicted_loss_percentage": prediction
            }
        )


# ============================================================
# 20. SAVE LOCAL EXPLANATIONS
# ============================================================

local_df = pd.DataFrame(
    local_rows
)


local_path = os.path.join(
    RESULTS_DIR,
    "local_xai_explanation.csv"
)


local_df.to_csv(
    local_path,
    index=False
)


print(
    "Local SHAP explanations saved to:"
)

print(
    local_path
)


# ============================================================
# 21. CREATE XAI SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULTS_DIR,
    "xai_summary.txt"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:


    f.write(
        "EXPLAINABLE AI - SHAP ANALYSIS\n"
    )


    f.write(
        "=" * 60 + "\n\n"
    )


    f.write(
        "Model: Random Forest Regressor\n"
    )


    f.write(
        f"Dataset rows: "
        f"{len(df)}\n"
    )


    f.write(
        f"Training rows: "
        f"{len(X_train)}\n"
    )


    f.write(
        f"Testing rows: "
        f"{len(X_test)}\n"
    )


    f.write(
        f"SHAP sample rows: "
        f"{sample_size}\n"
    )


    f.write(
        f"Encoded features: "
        f"{len(feature_names)}\n\n"
    )


    f.write(
        "TOP FEATURES BY SHAP IMPORTANCE\n"
    )


    f.write(
        "-" * 60 + "\n"
    )


    for _, row in importance_df.head(
        15
    ).iterrows():

        f.write(
            f"{row['feature']}: "
            f"{row['mean_abs_shap']:.6f}\n"
        )


    f.write(
        "\n"
    )


    f.write(
        "INTERPRETATION\n"
    )


    f.write(
        "-" * 60 + "\n"
    )


    f.write(
        "SHAP is used to explain how different "
        "input features influence the Random Forest "
        "prediction.\n\n"
    )


    f.write(
        "A larger absolute SHAP value means that the "
        "feature had a stronger influence on the "
        "prediction.\n\n"
    )


    f.write(
        "A positive SHAP value means the feature pushes "
        "the predicted food loss percentage higher.\n\n"
    )


    f.write(
        "A negative SHAP value means the feature pushes "
        "the predicted food loss percentage lower.\n"
    )


# ============================================================
# 22. FINISH
# ============================================================

print(
    "\nSummary saved to:"
)

print(
    summary_path
)


print(
    "\n" + "=" * 60
)

print(
    "EXPLAINABLE AI ANALYSIS COMPLETED SUCCESSFULLY."
)

print(
    "=" * 60
)


print(
    "\nFiles created:"
)

print(
    f"  {importance_path}"
)

print(
    f"  {global_plot_path}"
)

print(
    f"  {local_path}"
)

print(
    f"  {summary_path}"
)