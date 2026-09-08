# Movie Review Spoiler Classification

## Overview

Built a machine learning system to classify IMDb movie reviews into **Spoiler** and **Non-Spoiler** categories based on review text.

The project uses **TF-IDF text representation** with unigrams and bigrams and compares multiple traditional machine learning models. Model performance is evaluated using Accuracy, Precision, Recall, F1-score, ROC-AUC, and PR-AUC.

## Dataset

- Source: IMDb Movie Reviews Dataset
- Input: Movie review text
- Target: `is_spoiler`
  - `0` → Non-Spoiler
  - `1` → Spoiler
- A stratified working subset of up to 120,000 reviews was used for efficient experimentation.

## Methodology

1. Loaded and inspected IMDb review data.
2. Handled missing review text and checked duplicate records.
3. Cleaned text by:
   - Converting text to lowercase
   - Removing HTML tags
   - Removing URLs
   - Normalizing extra spaces
4. Performed stratified train-validation-test splitting.
5. Converted reviews into numerical features using **TF-IDF** with unigram and bigram features.
6. Compared four machine learning models:
   - Logistic Regression
   - Linear SVM
   - Multinomial Naive Bayes
   - XGBoost
7. Used **Truncated SVD** to reduce the high-dimensional sparse TF-IDF representation before XGBoost.
8. Evaluated models using Accuracy, Precision, Recall, F1-score, ROC-AUC, and PR-AUC.
9. Applied **5-fold Stratified Cross-Validation** to verify the stability of the Logistic Regression model.
10. Retrained the final model using the combined training and validation data and evaluated it once on the untouched test set.

## Cross-Validation Result

The Logistic Regression model achieved:

| Metric | Mean ± Std |
|---|---:|
| Accuracy | 0.7758 ± 0.0036 |
| Precision | 0.6385 ± 0.0125 |
| Recall | 0.3403 ± 0.0085 |
| F1-score | 0.4440 ± 0.0099 |
| ROC-AUC | 0.7585 ± 0.0037 |
| PR-AUC | 0.5611 ± 0.0051 |

The low standard deviations across the five folds indicate that the model performance is relatively stable across different subsets of the training data.

## Key Takeaways

- TF-IDF effectively converts movie review text into machine-learning features.
- Logistic Regression provided a strong baseline for spoiler classification.
- PR-AUC and F1-score were considered important evaluation metrics because accuracy alone does not fully describe classification performance.
- Stratified cross-validation was used to verify model stability.
- The final test set was kept untouched until the final evaluation.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
