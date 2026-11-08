# ClassifyLab — Breast Cancer Classification Dashboard

## a. Problem statement
Build and deploy an interactive machine-learning classification application that compares five classifiers on one common dataset. The application loads pre-trained models, accepts a labeled test CSV, computes six evaluation metrics, and displays a confusion matrix, classification report, ROC curve, and comparison charts. Models are **not retrained** when a user uploads test data.

## b. Dataset description
- **Dataset:** Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository.
- **Public source:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Instances:** 569.
- **Input features:** 30 numeric features.
- **Task:** Binary classification.
- **Target encoding:** `target = 0` means Benign; `target = 1` means Malignant.
- **Split:** Stratified 80% training / 20% hold-out test, `random_state=42`.
- The project includes `data/wdbc_dataset.csv` and the held-out `test_data.csv`.

## c. GitHub Repository Link
Add your public GitHub repository URL here after pushing this project:

**GitHub:** https://github.com/jeenaraju-sys/bits-ml-classification-streamlit

## d. Models used and evaluation

All five models are trained on the same training partition and evaluated on the same held-out test partition. The app recalculates these metrics for the selected uploaded CSV.

### Comparison table

| Model                  |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.9649 | 0.9960 |      0.9750 |   0.9286 | 0.9512 | 0.9245 |
| Decision Tree          |     0.9211 | 0.9448 |      0.9459 |   0.8333 | 0.8861 | 0.8299 |
| K-Nearest Neighbors    |     0.9561 | 0.9823 |      0.9744 |   0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes   |     0.9386 | 0.9934 |      1.0000 |   0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine |     0.9737 | 0.9947 |      1.0000 |   0.9286 | 0.9630 | 0.9442 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Excellent balanced performance (0.965 accuracy, 0.996 AUC, 0.951 F1). Precision is 0.975 and recall is 0.929, so it makes few false-positive predictions while detecting most malignant cases. |
| **Decision Tree** | Lowest overall scores on this split (0.921 accuracy, 0.886 F1, 0.830 MCC). It is easier to interpret, but the depth-limited tree misses more malignant cases (0.833 recall). |
| **K-Nearest Neighbors** | Strong performance (0.956 accuracy, 0.982 AUC, 0.938 F1). It offers a good balance, although its scores are slightly below Logistic Regression and SVM on this split. |
| **Gaussian Naive Bayes** | Achieves perfect precision (1.000) but lower recall (0.833), meaning its malignant predictions are reliable but it misses more actual malignant cases than the leading models. |
| **Support Vector Machine** | Best overall on this split: 0.974 accuracy, 0.963 F1, and 0.944 MCC. Precision is 1.000 and recall is 0.929, giving the strongest combined result among the five models. |

**Overall winner for this dataset:** **Support Vector Machine** on the supplied 20% hold-out test split, selected by F1 first, then AUC and MCC as tie-breakers. This is a split-specific result, not a guarantee of future performance.

### Metrics
- **Accuracy:** fraction of all predictions that are correct.
- **AUC:** area under the ROC curve, using model probability/decision scores.
- **Precision:** fraction of predicted malignant cases that are actually malignant.
- **Recall:** fraction of actual malignant cases correctly identified.
- **F1:** harmonic mean of precision and recall.
- **MCC:** correlation coefficient between observed and predicted classes.

## Application features
- CSV upload for labeled test data (30 required feature columns plus `target`).
- Left sidebar model selector and workspace navigation.
- Single-model evaluation and all-model comparison.
- Six metric cards, comparison table, interactive Plotly charts, ROC curve, confusion matrix, and classification report.
- Five saved `.joblib` model files. No training occurs in the deployed app.

## Run locally

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the dashboard
streamlit run app.py
```

Open the local URL printed by Streamlit, usually http://localhost:8501.

To recreate the saved models:

```bash
python train_models.py
```

## Deploy on Streamlit Community Cloud (FREE)

1. Create/sign in to GitHub and create a **public** repository.
2. Upload all project files and folders from this project. Ensure `app.py`, `requirements.txt`, `README.md`, `test_data.csv`, and `models/` are at the repository root.
3. Visit https://share.streamlit.io/ (or https://streamlit.io/cloud) and sign in with GitHub.
4. Choose **Create app / New app**, select your repository and `main` branch.
5. Set the main file path to `app.py`, then deploy.
6. Wait for the build to finish. Open the app URL and test the sidebar, comparison page, model metrics, charts, and CSV upload.
7. Copy the public app URL into your assignment PDF.

**Live Streamlit App:** https://bits-ml-classification-app.streamlit.app/

## Suggested repository structure

```text
classification_dashboard_project/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_comparison.csv
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
└── data/
    └── wdbc_dataset.csv
```

## Important submission notes
- Capture your **own screenshot** of the assignment running in BITS Virtual Lab. Do not use a mockup or a screenshot from another environment.
- Add the actual GitHub URL and deployed Streamlit URL to the final PDF.
- Include this README content in the final PDF, along with the BITS Virtual Lab screenshot.
- The included `test_data.csv` is the labeled hold-out test set used for the reported comparison metrics.

## Limitations and responsible use
This is an educational classification exercise using a public medical dataset. It is not a medical device and must not be used to diagnose, screen, or make clinical decisions.
