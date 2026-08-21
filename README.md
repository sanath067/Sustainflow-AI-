# SustainFlow AI – FAO Food Loss Prediction Module

## Project Overview

This module is a part of the SustainFlow AI project. It predicts the food loss percentage using the FAO Food Loss dataset. The prediction is based on the following inputs:

- Commodity
- Country
- Food Supply Stage
- Year

The module uses a Random Forest Regression model for prediction.

---

## Folder Structure

```
SustainFlow AI/
│
├── Codes/
│   ├── clean_fao_data.py
│   ├── augment_fao_data.py
│   ├── train_fao_model.py
│   └── hypothetical_demo.py
│
├── Datasets/
│   ├── Fao_raw.csv
│   ├── Fao_clean.csv
│   └── Fao_augmented.csv
│
├── Results/
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   ├── fao_model.joblib
│   ├── metrics.txt
│   └── hypothetical_demo_output.csv
│
├── requirements.txt
└── README.md
```

---

## Datasets Used

### 1. FAO Food Loss Dataset
Used for training the food loss prediction model. This is the only dataset currently integrated and trained on.

### 2. Walmart Sales Dataset (planned)
Not yet integrated. Referenced in the hypothetical end-to-end demo as a placeholder for a future demand-forecasting module.

### 3. Cold-Chain Sensor Dataset (planned)
Not yet integrated. Referenced in the hypothetical end-to-end demo as a placeholder for a future spoilage-prediction module.

---

## Steps to Run

Install the required libraries.

```bash
pip install -r requirements.txt
```

Run the programs in the following order.

```bash
python Codes/clean_fao_data.py
python Codes/augment_fao_data.py
python Codes/train_fao_model.py
python Codes/hypothetical_demo.py
```

---

## What Each Program Does

### clean_fao_data.py

- Reads the original FAO dataset.
- Removes duplicate records.
- Removes missing values.
- Keeps only the required columns.
- Saves the cleaned dataset.

---

### augment_fao_data.py

- Reads the cleaned dataset.
- Generates additional synthetic records.
- Adds small random variations to increase the dataset size.
- Saves the augmented dataset.

---

### train_fao_model.py

- Loads the augmented dataset.
- Trains a Random Forest Regression model.
- Evaluates the model using RMSE, MAE and R² Score.
- Saves graphs, metrics and the trained model.

---

### hypothetical_demo.py

This file demonstrates how the complete SustainFlow AI system can work.

It combines:

- Food loss prediction (FAO model — real, trained)
- Demand prediction (sample logic — placeholder for future Walmart Sales module)
- Spoilage prediction (sample logic — placeholder for future Sensor module)

Finally, it generates a recommendation based on all three predictions.

---

## Output Files

After running all programs, the following files are generated inside the Results folder.

- fao_model.joblib
- metrics.txt
- actual_vs_predicted.png
- feature_importance.png
- hypothetical_demo_output.csv

---