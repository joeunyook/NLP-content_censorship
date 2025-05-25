import pandas as pd

df = pd.read_csv("data/dataset_train_p_en.csv", engine="python", on_bad_lines='skip')
#print(df.head())
print(df[['full_text', 'censored']].head())

