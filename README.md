# TLDR

https://drive.google.com/drive/folders/1O-whG68wP_PJxDEEMKI7LxaYlEtcwSfB?usp=drive_link contains some data directories that are too large to push to the GitHub.

`data/dataset_train.csv`, `data/dataset_val.csv`, and `data/dataset_test.csv` are the most updated, original, and comprehensive data. These are used for model prediction.

Then I did a lot of messy preprocesses to generate `shap_data/beeswarm/dataset_test_uniq_en_processed.csv` from `data/dataset_test.csv`, and I used `shap_data/beeswarm/dataset_test_uniq_en_processed.csv` to generate shapley values and spreadsheets.



# Traceback of generating shapley values from my memory

First, I wrote some preprocessing codes for `data/dataset_test.csv` to generate `data/v_0305/dataset_test_uniq_en.csv`. I put this part of code in another server which I don't permission to access any more :(

`data/v_0305/dataset_test_uniq_en.csv` contains all `lang=='en'` rows from `data/v_0305/dataset_test_uniq.csv`.

For `data/v_0305/dataset_test_uniq.csv`, I did some processing steps here (found from `shapley_value.ipynb`..): 
```python
data['full_text_wo_label'] = data['full_text_wo_label'].apply(lambda x: x.strip() if type(x) != float else "")
data['full_text_wo_label']
data.to_csv('/nfs/turbo/coe-vvh/ljr/Censorship/data/v_0305/dataset_test_uniq.csv')
```

Data that I used to generate shapley values: `shap_data/beeswarm/dataset_test_uniq_en_processed.csv`.

It comes from `data/v_0305/dataset_test_uniq_en.csv` using the code below:
```python
data = pd.read_csv('/nfs/turbo/coe-vvh/ljr/Censorship/data/v_0305/dataset_test_uniq_en.csv', engine='python', index_col=0)
data['full_text_wo_label'] = data['full_text_wo_label'].str.replace(r'[^\w\s#]', '', regex=True).str.strip()
data = data.loc[data['full_text_wo_label'].str.contains(' ')]
data.to_csv("/nfs/turbo/coe-vvh/ljr/Censorship/shap_data/beeswarm/dataset_test_uniq_en_processed.csv")
```

Generate shapley values:
```python

from tqdm import tqdm
# define a prediction function
def f(x):
    tv = torch.tensor([tokenizer.encode(v, pad_to_max_length=True, max_length=500) for v in x]).cuda()
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    # val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    # modify val
    val = scores[:, 1]
    return val

tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
model = transformers.AutoModelForSequenceClassification.from_pretrained("roberta-base").cuda()

model.load_state_dict(torch.load(
    '/nfs/turbo/coe-vvh/ljr/Censorship/models/20230311-10:45:13_binary_0.1_8_1e-05_adamw.pt'))

explainer = shap.Explainer(f, shap.maskers.Text())

# data = data[data['full_text_wo_label'].str.contains(' ')]
```


```python
withheld_list = [None] # 'TR', 'RU', 
# withheld_list = ['TR', 'RU', 'DE', 'FR', 'IN']
# withheld_list = ['DE']
# withheld_list = [None]
# sample 500



for withheld in tqdm(withheld_list):
    flag = 1        
    new_data = 0
    if withheld is not None:
        new_data = data[data['withheld'].fillna('').str.contains(f"'{withheld}")]
    else:
        new_data = data
    new_data.to_csv(f'/nfs/turbo/coe-vvh/ljr/Censorship/shap_data/beeswarm/v_0328_dataset_test_uniq_en_{str(withheld)}.csv')
    if withheld is not None:
        if withheld == 'RU':
            new_data = new_data.sample(n=400, replace=False, random_state=1)
        elif withheld == 'DE':
            new_data = new_data.sample(n=500, replace=False, random_state=1)
        elif withheld == 'IN':
            new_data = new_data
        else:
            new_data = new_data.sample(n=500, replace=False, random_state=1) # for test
    else:
        new_data = new_data.sample(n=5000, replace=False, random_state=1)

    shap_values_bert = explainer(new_data['full_text_wo_label'].values.tolist())

    with open(f'/nfs/turbo/coe-vvh/ljr/Censorship/shap_data/beeswarm/v0328_shap_vals_test_uniq_en_{str(withheld)}_samp500.pkl', 'wb') as f:
        pickle.dump(shap_values_bert, f)
    shap.plots.bar(shap_values_bert.mean(0), max_display=50)
    
            
#     shap.plots.beeswarm(shap_values)

```


Build `shap_data/spreadsheet/union_features_with_ranks.csv` from `shap_data/beeswarm/v0328_shap_vals_test_uniq_en_{str(withheld)}_samp500.pkl`: 

```python
from shapley_value import build_spreadsheet
spread_df = build_spreadsheet(['None', "TR", 'DE', 'FR', "RU", 'IN'])
```


From `shap_data/spreadsheet/union_features_with_ranks.csv` to `shap_data/spreadsheet/union_features_with_ranks_top.csv`:

```python
from shapley_value import build_spreadsheet
spread_df = build_spreadsheet(['None', "TR", 'DE', 'FR', "RU", 'IN'], top=500)
```

# Model training: 

- `log/` contains my previous training logs.
- `models/` stores those model checkpoints.
- `output/` contains model prediction outputs.