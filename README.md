# Confidence-Driven Inference Tutorial

## What this tutorial does
The Jupyter notebook walks you through an **end-to-end, automated human-AI annotation pipeline** based on **Confidence-Driven Inference (CDI)**. We collect human annotations via **Prolific** or **Amazon Mechanical Turk**, and LLM annotations through **OpenAI**'s API. The example is based on the method introduced in:

```
Can Unconfident LLM Annotations Be Used for Confident Conclusions? Kristina Gligorić*, Tijana Zrnic*, Cinoo Lee*, Emmanuel Candès, and Dan Jurafsky. NAACL, 2025.  
```
The goal is to estimate a target statistic about a text corpus while **minimizing costly human labels** by:

1. **Obtaining cheap-but-noisy labels from an LLM** together with its confidence scores.  
2. **Collecting a small sample of human labels** to calibrate.  
3. **Iteratively sampling the most informative texts** for additional human annotation using the CDI sampling rule.  
4. **Computing the final point estimate & bootstrap confidence interval** once the annotation budget is exhausted.

We focus on annotating texts for **politeness** and showcase the estimation of **two target statistics**:
1. $mean(H)$: prevalence of politeness, i.e., the fraction of texts in the corpus that are polite.
2. $\beta_{hedge}$: the impact of linguistic features of hedging ($X$) on the perceived politeness ($H$), estimated with a logistic regression.

Although the example focuses on detecting *politeness* and estimating these two target statistics, you can adapt the flow to any text classification task and any other target statistic.

---

## Notebook outline

| Section | Purpose |
|---------|---------|
| **Import libraries** | Loads scientific stack (`numpy`, `scipy`, `pandas`, `tqdm`, Qualtrics/Prolific/MTURK helpers, and `openai` for LLM calls). |
| **Parameter blocks** | Separate cells let you tune *CDI hyper-parameters*, *LLM sampling settings*, and *human‑annotation settings* (batch size, budget, etc.). |
| **Step&nbsp;1 – LLM annotation** | Loads a CSV of raw texts (`data/politeness_dataset.csv`), queries the LLM for a label & confidence for each row, and stores results in the working `data` frame. |
| **Step&nbsp;2 – Initial human labels** | Publishes the first batch of texts to Prolific or MTURK, waits for responses, and merges them back into `data`. Initialize the sampling rule to obtain per‑item selection probabilities. |
| **Step&nbsp;3 – Iterative sampling loop** | For each batch: choose texts with highest CDI scores → post new survey → ingest responses → update CDI state. |
| **Step&nbsp;4 – Estimation** | After the last batch, calculate the CDI estimator and a bootstrap 90 % confidence interval. Timing information for the whole pipeline is also logged. |


---

## Repository structure

```
project/
├── tutorial_version_1.ipynb
├── tutorial_version_2.ipynb
├── tutorial_version_3.ipynb
├── tutorial_version_4.ipynb
├── tutorial_version_5.ipynb
├── data/
│   └── politeness_dataset.csv
├── utils/                  # helper modules (e.g., survey API wrappers, inference modules)
|── requirements.txt
|── credentials.txt
└── README.md               
```

---

## Tutorial Notebooks

This repository includes five versions of a tutorial:

1. **`tutorial_version_1_high-level_pre-collected_labels.ipynb`**  
   Adaptive label collection using high-level functions and pre-collected human labels (for demonstration purposes).

2. **`tutorial_version_2_high-level_collecting_labels.ipynb`**  
   Adaptive label collection using high-level functions, with live collection of a small set of new human and LLM labels.

3. **`tutorial_version_3_low-level_pre-collected_labels.ipynb`**  
   Adaptive label collection using low-level code, with pre-collected human labels.

4. **`tutorial_version_4_low-level_collecting_labels.ipynb`**  
   Adaptive label collection using low-level code, with live collection of a small set of new human and LLM labels.

5. **`tutorial_version_5_non-adaptive.ipynb`**  
   Non-adaptive label collection on a random sample of texts, using pre-collected human labels.

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
- Printed estimate and 90 % CI for the target metric.

---

## Adapting the tutorial

- Swap in your own dataset with **`Text`** and **feature column(s)**.  
- Update the `mapping_categories` dict to match your label set.  
- Tweak `burnin_steps`, `batch_size`, and `budget` to suit annotation cost constraints.  
- Plug in a different LLM prompt or model name to target alternative tasks.

---

## References

Can Unconfident LLM Annotations Be Used for Confident Conclusions? Kristina Gligorić*, Tijana Zrnic*, Cinoo Lee*, Emmanuel Candès, and Dan Jurafsky. NAACL, 2025.  

This tutorial is presented as part of the International Conference on Computational Social Science (IC2S2) tutorial on Bridging Human and LLM Annotations for Statistically Valid Computational Social Science. URL: https://sites.google.com/view/ic2s2-bridging-human/home 

