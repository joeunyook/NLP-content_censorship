import pandas as pd
import ast
from sklearn.metrics import classification_report

# Load the prediction CSV
df = pd.read_csv("output/roberta-edl-country_binary_0.1_8_1e-05_adamw.csv")

# Parse 'withheld' column to extract country codes
def extract_countries(val):
    try:
        d = ast.literal_eval(val)
        return d.get("country_codes", [])
    except:
        return []

df["country_list"] = df["withheld"].apply(extract_countries)

# Explode to one row per country
df_exploded = df.explode("country_list")

# Countries of interest
target_countries = ["DE", "TR", "IN", "RU", "FR"]

# Prepare report
report_output = []
report_output.append("Country-wise F1 Score Report (Only DE, TR, IN, RU, FR)\n" + "=" * 40 + "\n")

for country in target_countries:
    subset = df_exploded[df_exploded["country_list"] == country]
    
    if subset.empty:
        continue
    
    y_true = subset["y_truth"]
    y_pred = subset["y_pred"]

    report_output.append(f"\n Country: {country} (n={len(subset)})\n")
    report_output.append(classification_report(y_true, y_pred, digits=4))
    report_output.append("=" * 40 + "\n")

# Save to txt file
output_path = "output/edl_country_f1_report.txt"
with open(output_path, "w") as f:
    f.write("\n".join(report_output))

print(f"Report saved to {output_path}")

