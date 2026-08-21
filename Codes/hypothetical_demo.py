import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("Results", "fao_model.joblib")

# PART 1 — REAL: load the trained FAO model and predict loss_percentage

def predict_real_loss(commodity, country, stage, year):
    model = joblib.load(MODEL_PATH)
    X_new = pd.DataFrame([{
        "commodity": commodity,
        "country": country,
        "food_supply_stage": stage,
        "year": year,
    }])
    predicted_loss_pct = model.predict(X_new)[0]
    return round(predicted_loss_pct, 2)

# PART 2 — HYPOTHETICAL: placeholder demand signal
# (stands in for a future ML demand-forecasting module trained on
#  real sales data such as Walmart/M5)

def hypothetical_demand_signal(recent_sales_units, stock_on_hand_units):
    ratio = stock_on_hand_units / max(recent_sales_units, 1)
    if ratio > 1.5:
        return "Low", ratio
    elif ratio > 0.8:
        return "Normal", ratio
    else:
        return "High", ratio

# PART 3 — HYPOTHETICAL: placeholder spoilage risk
# (stands in for a future sensor-based classifier trained on Sensor.csv)

def hypothetical_spoilage_risk(temperature_c, humidity_pct):
    if temperature_c > 8 or humidity_pct > 90:
        return "High"
    elif temperature_c > 4 or humidity_pct > 80:
        return "Medium"
    else:
        return "Low"



# COMBINE all three signals into one recommendation

def generate_recommendation(loss_pct, demand_level, spoilage_risk):
    reasons = []
    action = "Continue normal handling"

    if spoilage_risk == "High" or loss_pct > 20:
        action = "URGENT: Prioritize dispatch / discount sale today"
        reasons.append(f"High spoilage risk ({spoilage_risk}) and/or high predicted loss ({loss_pct}%)")
    elif demand_level == "Low" and loss_pct > 10:
        action = "Reduce next restock order, monitor closely"
        reasons.append(f"Low demand + moderate predicted loss ({loss_pct}%) -> overstock risk")
    elif spoilage_risk == "Medium":
        action = "Increase monitoring frequency"
        reasons.append("Medium spoilage risk detected from sensor readings")
    else:
        reasons.append("All signals within normal range")

    return action, reasons



# EXAMPLE SCENARIO 

def run_example():
    print("=" * 60)
    print("SustainFlow AI - Hypothetical End-to-End Demo")
    print("=" * 60)

    #  Example inputs 
    commodity = "Tomatoes"
    country = "Cambodia"
    stage = "Post-harvest"
    year = 2024
    recent_sales_units = 400
    stock_on_hand_units = 650
    temperature_c = 9.5
    humidity_pct = 88

    print(f"\nScenario: {commodity} in {country}, stage = {stage}, year = {year}")
    print(f"Sales/stock: recent_sales={recent_sales_units} units, "
          f"stock_on_hand={stock_on_hand_units} units")
    print(f"Sensor readings: temperature={temperature_c}C, humidity={humidity_pct}%")

    # Load trained FAO model
    loss_pct = predict_real_loss(commodity, country, stage, year)
    print(f"\n[REAL MODEL] Predicted FAO loss percentage: {loss_pct}%")

    # Load trained FAO model
    demand_level, ratio = hypothetical_demand_signal(recent_sales_units, stock_on_hand_units)
    print(f"[HYPOTHETICAL] Demand signal: {demand_level} (stock/sales ratio = {ratio:.2f})")

    # Temporary spoilage prediction logic
    spoilage_risk = hypothetical_spoilage_risk(temperature_c, humidity_pct)
    print(f"[HYPOTHETICAL] Spoilage risk: {spoilage_risk}")

    # Combine into a recommendation
    action, reasons = generate_recommendation(loss_pct, demand_level, spoilage_risk)
    print(f"\n>>> RECOMMENDATION: {action}")
    for r in reasons:
        print(f"    - {r}")

    result_row = {
        "commodity": commodity, "country": country, "stage": stage, "year": year,
        "recent_sales_units": recent_sales_units, "stock_on_hand_units": stock_on_hand_units,
        "temperature_c": temperature_c, "humidity_pct": humidity_pct,
        "predicted_loss_pct_REAL_MODEL": loss_pct,
        "demand_signal_HYPOTHETICAL": demand_level,
        "spoilage_risk_HYPOTHETICAL": spoilage_risk,
        "recommendation": action,
    }
    out_path = os.path.join("Results", "hypothetical_demo_output.csv")
    pd.DataFrame([result_row]).to_csv(out_path, index=False)
    print(f"\nSaved this run to: {out_path}")


if __name__ == "__main__":
    run_example()