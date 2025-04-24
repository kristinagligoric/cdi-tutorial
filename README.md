# README — Confidence-Driven Inference Tutorial

## What this notebook does
This Jupyter notebook walks you through an **end-to-end, automated human-AI annotation pipeline** based on **Confidence-Driven Inference (CDI)**. The example is based on the method introduced in:

```
Can Unconfident LLM Annotations Be Used for Confident Conclusions? Kristina Gligorić*, Tijana Zrnic*, Cinoo Lee*, Emmanuel Candès, and Dan Jurafsky. NAACL, 2025.  
```
The goal is to estimate a target statistic about a text corpus while **minimizing costly human labels** by:

1. **Obtaining cheap-but-noisy labels from an LLM** together with its confidence scores.  
2. **Collecting a small sample of human labels** to calibrate.  
3. **Iteratively sampling the most informative texts** for additional human annotation using the CDI sampling rule.  
4. **Computing the final point estimate & bootstrap confidence interval** once the annotation budget is exhausted.

Although the example focuses on detecting *politeness* in a corpus, you can adapt the flow to any binary or multi-class text classification task.

---

## Notebook outline

| Section | Purpose |
|---------|---------|
| **Import libraries** | Loads scientific stack (`numpy`, `scipy`, `pandas`, `tqdm`, Qualtrics/Prolific helpers, and `openai` for LLM calls. |
| **Parameter blocks** | Separate cells let you tune *CDI hyper-parameters*, *LLM sampling settings*, and *human‑annotation settings* (batch size, budget, etc.). |
| **Step&nbsp;1 – LLM annotation** | Loads a CSV of raw texts (`data/politeness_dataset.csv`), queries the LLM for a label & confidence for each row, and stores results in the working `data` frame. |
| **Step&nbsp;2 – Initial human labels** | Publishes the first batch of texts to Qualtrics/Prolific, waits for responses, and merges them back into `data`. Initialize the sampling rule to obtain per‑item selection probabilities. |
| **Step&nbsp;3 – Iterative sampling loop** | For each batch: choose texts with highest CDI scores → post new survey → ingest responses → update CDI state. |
| **Step&nbsp;4 – Estimation** | After the last batch, calculate the CDI estimator and a bootstrap 95 % confidence interval. Timing information for the whole pipeline is also logged. |


---

## Repository structure

```
project/
├── tutorial.ipynb          # tutorial notebook
├── data/
│   └── politeness_dataset.csv
├── utils/                  # helper modules (e.g., survey API wrappers, inference modules)
|── requirements.txt
└── README.md               
```

---

## Requirements & setup

1. **Python ≥ 3.9**  
2. Install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## How to run

```bash
jupyter notebook
# open tutorial.ipynb and run cells top‑to‑bottom
```

- **Dry‑run mode:** Keep `COLLECT_LLM = False` and `COLLECT_HUMAN = False` in the parameter cells to skip external API calls while you familiarize yourself with the flow.  

---

## Expected outputs

- Console log showing batch progression and total wall‑clock time.  
- Printed estimate and 95 % CI for the target metric.

---

## Adapting the tutorial

- Swap in your own dataset with **`Text`** and **feature column(s)**.  
- Update the `mapping_categories` dict to match your label set.  
- Tweak `burnin_steps`, `batch_size`, and `budget` to suit annotation cost constraints.  
- Plug in a different LLM prompt or model name to target alternative tasks.

---

## References

Can Unconfident LLM Annotations Be Used for Confident Conclusions? Kristina Gligorić*, Tijana Zrnic*, Cinoo Lee*, Emmanuel Candès, and Dan Jurafsky. NAACL, 2025.  

