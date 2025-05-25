import subprocess

command = [
    "python", "train.py",
    "--pretrained_model", "roberta-base",
    "--dataset_df_dir", "data/",
    "--splits_filename",
    "dataset_train_p_en.csv",
    "dataset_val_p_en.csv",
    "dataset_test_p_en.csv",
    "--text_col", "full_text_proc",
    "--y_col", "censored",
    "--output_type", "binary",
    "--batch_size", "8",
    "--lr", "1e-5",
    "--dropout_rate", "0.1",
    "--n_epochs", "3",
    "--log_dir", "log/",
    "--model_save_dir", "models/"
]

result = subprocess.run(command, capture_output=True, text=True)

print("=== STDOUT ===")
print(result.stdout)
print("=== STDERR ===")
print(result.stderr)
