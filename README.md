## Project Origin

This repository is based on the original censorship dataset and task setup from:

> **"Revealing Hidden Mechanisms of Cross-Country Content Moderation with Natural Language Processing"**  
> [Yadav et al., ACL 2024]( http://arxiv.org/abs/2503.05280)

I reused their dataset and adapted it for a **Uncertainty tracking enabled censorship classification task**.

### My Contributions
This version of the repo extends the original work by:

- Replacing the softmax classification head with an **Evidential Deep Learning (EDL) head** to model belief and uncertainty.
- Applying EDL to Twitter censorship detection for more **interpretable and calibrated binary decisions**.
- Selecting **borderline cases with high uncertainty** and injecting **LLM-generated context at inference** to test changes in model behavior.
- Providing statistical evaluation of how LLM context affects trust and calibration.

These experiments were conducted as part of my research and are summarized in the paper:

📄 **[Towards Trustworthy Censorship Detection: Evidential Deep Learning and LLM Context in High-Uncertainty NLP Decisions (PDF)](https://github.com/joeunyook/NLP-content_censorship/blob/main/NLP_paper_final.pdf)**  
✍ *Joeun Yook, University of Toronto*

---

# Model training: 

- `log/` contains my previous training logs.
- `models/` stores those model checkpoints.
- `output/` contains model prediction outputs.
