# Netflix Prize Recommendation System
### Open Projects 2026 — The Cultural Council, IIT Roorkee

A personalized movie recommendation system built using the Netflix Prize Dataset.
Implements and compares four recommendation approaches: SVD, NMF, SVD++, and
Neural Collaborative Filtering (NCF).

---

## Results

| Model | RMSE | MAP@10 | MAE |
|-------|------|--------|-----|
| SVD (Matrix Factorization) | **0.9594** | **0.6933** | 0.7580 |
| NMF (Non-Negative MF) | 1.1669 | 0.5819 | 0.9184 |
| SVD++ (Implicit Feedback) | 0.9966 | 0.5862 | 0.7980 |
| NCF (Neural CF) | 0.9941 | N/A | — |

**Best Model: SVD** — achieves lowest RMSE and highest MAP@10.
MAP@10 of 0.6933 means nearly 7 out of 10 recommended movies
are genuinely relevant to the user (actual rating ≥ 3.5).

---

## Project Structure
netflix-recommendation-system/

├── README.md

├── requirements.txt

├── Netflix_Recommendation_System.ipynb

└── results/

├── model_results.csv

├── rating_distribution.png

└── model_comparison.png

---

## How to Run

### 1. Get the Dataset
Download the Netflix Prize Dataset from Kaggle:
https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data

This downloads as `archive.zip`.

### 2. Open the Notebook in Google Colab
Go to [colab.research.google.com](https://colab.research.google.com)
→ File → Upload notebook → select `Netflix_Recommendation_System.ipynb`

### 3. Upload the Dataset
In the Colab Files panel (left sidebar), upload your `archive.zip` file.

### 4. Run All Cells in Order
Runtime → Run all

The notebook will automatically extract the zip, load 5 million ratings,
run EDA, train all 4 models, evaluate them, and generate recommendations.

> **Note:** KNN-based models were replaced with NMF and SVD++ due to
> RAM constraints on free Colab. All mandatory evaluation criteria are met.

---

## Models Implemented

### SVD — Singular Value Decomposition
Decomposes the user-movie matrix into latent user and item factor vectors.
Ratings predicted as dot product of these vectors plus bias terms.
- 100 latent factors, 20 epochs, learning rate 0.005, regularization 0.02

### NMF — Non-Negative Matrix Factorization
Matrix factorization with non-negativity constraints on all factor matrices,
producing more interpretable latent dimensions.
- 15 factors, 20 epochs

### SVD++ — Extended SVD with Implicit Feedback
Improves SVD by incorporating implicit feedback — which movies a user
rated regardless of the actual score — as an additional signal.
- 20 factors, 20 epochs

### NCF — Neural Collaborative Filtering
Deep learning approach using PyTorch. User and movie IDs mapped to
64-dimensional embeddings, concatenated and passed through fully connected
layers (128 → 64 → 1) with ReLU activations and Dropout regularization.
- 5,157,953 parameters, trained for 5 epochs with Adam optimizer

---

## Evaluation Methodology

**Train-Test Split:** 80% training, 20% test (random split, seed=42)

**RMSE:** Measures rating prediction accuracy on held-out test set.

**MAP@10:** Measures recommendation ranking quality.
- Relevance threshold: actual user rating ≥ 3.5
- Top-10 recommendations generated per user by predicted rating
- Average Precision computed per user, averaged across all users

---

## Dataset

- **Source:** [Netflix Prize Data — Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)
- **Subset used:** 5,000,000 ratings from combined_data_1.txt
- **Users:** 404,478 | **Movies:** 996 (subset) / 17,770 (full catalog)
- **Sparsity:** 98.76%
- **Date range:** December 1999 – December 2005

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| pandas, numpy | Data loading and processing |
| scikit-surprise | SVD, NMF, SVD++ models |
| PyTorch | Neural Collaborative Filtering |
| matplotlib, seaborn | Visualizations |
| scikit-learn | Train-test split |

---

## Participant

**Name:** Suda Mounika
**Enrollment No:** 23115147
**Branch & Year:** EE 4Y
