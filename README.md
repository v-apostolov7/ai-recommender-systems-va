# Movie Recommender System (Collaborative Filtering)

A Machine Learning project evaluating collaborative filtering algorithms on movie ratings data, featuring an interactive Streamlit dashboard to compare real user ratings with SVD model predictions.

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
recommender_systems_inteligent_systems/
├── database/
│   ├── ratings.csv            # Excluded: Original dataset (~740 MB)
│   └── ratings_sample.csv     # Lightweight sample for testing & demo
├── src/
│   ├── data_loader.py         # Dataset sampling and Surprise Reader config
│   ├── models.py              # Dictionary registry of benchmark models
│   └── train_eval.py          # Cross-validation & RMSE evaluation logic
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
git clone https://github.com/v-apostolov7/recommender_systems_inteligent_systems.git
cd recommender_systems_inteligent_systems

### 2. Set up virtual environment & install dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

> **Note on scikit-surprise installation:** If running on Windows, ensure C++ Build Tools are installed, or install via pre-built wheels if pip encounters compilation errors.

### 3. Run the Streamlit Dashboard
streamlit run app.py

---

## Evaluation & SVD Behavior

The production inference pipeline uses the **SVD (Singular Value Decomposition)** model. In collaborative filtering:
* Ratings generally span the range [0.5, 5.0].
* Average absolute residuals between predicted and actual ratings typically hover between **0.8 and 1.2**, consistent with benchmark MovieLens performance on sparse matrices.