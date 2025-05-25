import pandas as pd

# Load the EDL output CSV
df = pd.read_csv('data/test_10_edl_context_augmented.csv')

# Add model prediction column: 0 if belief_0 > 0.5, else 1
df['model_prediction'] = df['belief_0'].apply(lambda x: 0 if x > 0.5 else 1)

# Save the updated CSV
df.to_csv('data/test_10_edl_context_augmented_with_prediction.csv', index=False)
