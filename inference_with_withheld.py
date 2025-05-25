import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import ast

# ---- CONFIG ----
MODEL_PATH = "models/20250512-08:02:16_binary_0.1_8_1e-05_adamw.pt"
TEST_CSV_PATH = "data/dataset_test_p_en.csv"
OUTPUT_CSV_PATH = "output/roberta-base-country.csv"
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- LOAD MODEL ----
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---- LOAD TEST SET ----
df = pd.read_csv(TEST_CSV_PATH)
texts = df["text_proc"].astype(str).tolist()
labels = df["censored"].tolist()
withheld = df["withheld"].tolist()

# ---- TOKENIZE ----
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
input_ids = encodings["input_ids"].to(DEVICE)
attention_mask = encodings["attention_mask"].to(DEVICE)

# ---- INFERENCE ----
predictions = []
with torch.no_grad():
    for i in range(0, len(input_ids), BATCH_SIZE):
        input_batch = input_ids[i:i+BATCH_SIZE]
        mask_batch = attention_mask[i:i+BATCH_SIZE]
        output = model(input_batch, attention_mask=mask_batch)
        pred = torch.argmax(output.logits, dim=1)
        predictions.extend(pred.cpu().numpy())

# ---- EXTRACT COUNTRY CODES  ----
def extract_country(val):
    try:
        obj = ast.literal_eval(val)
        return obj.get("country_codes", [None])[0]
    except:
        return None

country_codes = list(map(extract_country, withheld))

# ---- SAVE RESULTS ----
output_df = pd.DataFrame({
    "y_truth": labels,
    "y_pred": predictions,
    "withheld": withheld,
    "country": country_codes
})

output_df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"\n Saved: {OUTPUT_CSV_PATH}")
print(output_df["country"].value_counts())
