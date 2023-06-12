import transformers
import torch
import numpy as np
import scipy as sp
from shap_customized import shap
import pdb
np.random.seed(seed=0)

# load a BERT sentiment analysis model
tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")


# print(tokenizer)

model = transformers.AutoModelForSequenceClassification.from_pretrained("roberta-base")

model.load_state_dict(torch.load(
    '/nfs/turbo/coe-vvh/ljr/Censorship/models/20230219-03:24:33_binary_0.1_8_1e-05_adamw.pt', map_location=torch.device("cpu")))

# define a prediction function
def f(x):
    tv = torch.tensor([tokenizer.encode(v, pad_to_max_length=True, max_length=500) for v in x])
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    # val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    # modify val
    val = scores[:, 1]
    return val

# build an explainer using a token masker
explainer = shap.Explainer(f, shap.maskers.Text())

# explain the model's predictions on IMDB reviews
# shap_values = explainer(tmp_df['full_text_proc'].values.tolist())

pdb.set_trace()
shap_values = explainer(["#OccupyOakland: footage shows police", "Just voted Ken."])
pdb.set_trace()
shap.plots.text(shap_values)
