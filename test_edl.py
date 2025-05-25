import pandas as pd
import torch
from transformers import AutoTokenizer
from models import Model
import argparse
from tqdm import tqdm
import os
import google.generativeai as genai

# --- 1. Load args ---
args = argparse.Namespace()
args.pretrained_model = 'roberta-base'
args.output_type = 'binary'
args.num_classes = 2
args.num_numeric_features = 0
args.use_edl = True
args.dropout_rate_curr = 0.1
args.feature_size = 768
args.hidden_dim_curr = None
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. Load model ---
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
model = Model(args)
model.load_state_dict(torch.load("models/20250516-08:48:11_binary_0.1_8_1e-05_adamw.pt"))
model.to(args.device)
model.eval()

# --- 3. Load data (only first 10 rows) ---
df = pd.read_csv("data/filtered_test_data_uncertainty_gt_0.95.csv")
texts = df['text_proc'].tolist()

# --- 4. Init Gemini ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")

# --- 5. Get context for each text ---
def get_context(tweet):
    prompt = f"""In under 50 words, objectively explain the social, cultural, or political context relevant to the content of this specific tweet. Focus solely on the tweet's context.

Tweet: \"{tweet}\"
Explanation:"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return ""

print("Generating Gemini context...")
context_augmented_texts = []
for text in tqdm(texts):
    context = get_context(text)
    combined = context + " " + text  # fused input: [context + tweet]
    context_augmented_texts.append(combined)

# --- 6. Inference ---
belief_0, belief_1, uncertainty, evidence = [], [], [], []

print("Running EDL inference...")
with torch.no_grad():
    tokens = tokenizer(context_augmented_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    input_ids = tokens['input_ids'].to(args.device)

    alpha = model(input_ids)
    S = alpha.sum(dim=1, keepdim=True)
    belief = alpha / S
    unc = args.num_classes / S
    evid = S - args.num_classes

    belief_0.extend(belief[:, 0].cpu().tolist())
    belief_1.extend(belief[:, 1].cpu().tolist())
    uncertainty.extend(unc[:, 0].cpu().tolist())
    evidence.extend(evid[:, 0].cpu().tolist())

# --- 7. Save output ---
df['gemini_context'] = context_augmented_texts
df['belief_0'] = belief_0
df['belief_1'] = belief_1
df['uncertainty'] = uncertainty
df['evidence'] = evidence
df.to_csv("output/test_edl_context_augmented.csv", index=False)

print("Saved updated EDL predictions to output/test_edl_context_augmented.csv")
