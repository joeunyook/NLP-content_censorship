import pandas as pd
from sklearn.metrics import classification_report

# Load output predictions
df = pd.read_csv("output/roberta-base-country.csv")

# Drop missing or malformed country entries
df = df.dropna(subset=["country"])
df = df[df["country"].apply(lambda x: isinstance(x, str) and len(x) == 2)]

# Only keep the 5 countries from the paper
valid_countries = ["DE", "TR", "IN", "RU", "FR"]
df = df[df["country"].isin(valid_countries)]

# Prepare output file
report_file = "country_f1_report.txt"
with open(report_file, "w") as f:
    f.write("Country-wise F1 Score Report (Only DE, TR, IN, RU, FR)\n")
    f.write("=" * 40 + "\n")

    for country in df["country"].unique():
        subset = df[df["country"] == country]
        report = classification_report(subset["y_truth"], subset["y_pred"], digits=4)

        header = f"\n Country: {country} (n={len(subset)})\n"
        print(header)
        print(report)

        f.write(header)
        f.write(report)
        f.write("\n" + "=" * 40 + "\n")

print(f"\n done : saved filtered country-wise F1 report to: {report_file}")
