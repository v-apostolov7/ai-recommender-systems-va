# Movie Recommender System (Collaborative Filtering)

A Machine Learning project evaluating collaborative filtering algorithms on movie ratings data, featuring an interactive Streamlit dashboard to compare real user ratings with SVD model predictions.

---

## Live demo
[Click here for live demo](https://ai-recommender-systems-va.streamlit.app/)

---

## Project Overview

This repository demonstrates matrix factorization and collaborative filtering techniques using the **Surprise** library. The project benchmarks baseline models against **Singular Value Decomposition (SVD)** and **k-Nearest Neighbors (KNN)** using Cross-Validation RMSE, followed by an interactive inspection dashboard.

### Key Highlights
* **Algorithm Comparison:** Evaluates NormalPredictor (baseline), KNNBasic, and SVD via 3-fold cross-validation.
* **Pre-trained Inference:** Stores serialized model artifacts (model.pkl) for rapid predictions without retraining.
* **Interactive Dashboard:** Streamlit UI allowing dynamic inspection of arbitrary user histories against predicted scores.

---

## Repository Architecture

```
ai-recommender-systems-va/
├── database/
│   ├── ratings.csv            # Excluded: Original dataset from MovieLens (~740 MB)
│   └── ratings_sample.csv     # Lightweight sample for testing & demo
├── src/
│   ├── data_loader.py         # Dataset sampling and Surprise Reader config
│   ├── models.py              # Dictionary registry of benchmark models
│   └── train_eval.py          # Cross-validation & RMSE evaluation logic
├── main.py                    # Comparison of the 3 models and generating of model.pkl
├── app.py                     # Streamlit comparison dashboard
├── model.pkl                  # Serialized trained SVD model
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Tech Stack

* **Language:** Python 3.10+
* **ML / Recommender Library:** scikit-surprise
* **Data Processing:** pandas
* **Web UI:** Streamlit

---

## Getting Started

### 1. Clone the repository
git clone https://github.com/v-apostolov7/ai-recommender-systems-va.git

cd ai-recommender-systems-va

### 2. Set up virtual environment & install dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

> **Note on scikit-surprise installation:** If running on Windows, ensure C++ Build Tools are installed, or install via pre-built wheels if pip encounters compilation errors.

### 3. Run the Streamlit Dashboard
streamlit run app.py

---

## Evaluation & SVD Behavior

The production inference pipeline uses the **SVD (Singular Value Decomposition)** model, which is benchmarked against **KNNBasic** and a baseline **NormalPredictor**. 

**Data Processing & Sampling:**
To optimize memory usage and ensure rapid cross-validation, the dataset is randomly sampled down to **100,000 records** (`random_state=42`) prior to model training.

**Performance Metrics:**
- All collaborative filtering predictions are constrained to the official rating scale of **[0.5, 5.0]**.
- Models are evaluated using **3-fold Cross-Validation** with **RMSE (Root Mean Square Error)** as the primary metric.
- Average absolute residuals between predicted and actual SVD ratings typically hover between **0.8 and 1.2**, consistent with benchmark MovieLens performance on sparse matrices.