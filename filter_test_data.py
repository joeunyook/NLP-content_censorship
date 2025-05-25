import pandas as pd

# Load the EDL output CSV and original test dataset
edl_df = pd.read_csv('output/roberta-edl-country_binary_0.1_8_1e-05_adamw.csv')
test_df = pd.read_csv('data/dataset_test_p_en.csv')

# Ensure indices align (assuming `data_index` in EDL output corresponds to row index in test set)
# Filter rows with uncertainty > 0.975
high_uncertainty = edl_df[edl_df['uncertainty'] > 0.975]

# Use the data_index column to select corresponding rows from the original test set
filtered_test_df = test_df.iloc[high_uncertainty['data_index'].values]

# Optionally, reset index and save
filtered_test_df = filtered_test_df.reset_index(drop=True)
filtered_test_df.to_csv('data/filtered_test_data_uncertainty_gt_0.95.csv', index=False)



