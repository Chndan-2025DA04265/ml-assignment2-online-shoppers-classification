# ML Assignment 2 - Online Shoppers Purchasing Intention

## a. Problem Statement

The goal is to predict whether a visitor to an online shopping website will make a purchase or not based on their browsing session data. This is a binary classification problem where we predict the 'Revenue' column (True = purchase made, False = no purchase). We implemented 5 different ML models and compared their performance.

## b. Dataset Description

- **Name:** Online Shoppers Purchasing Intention Dataset
- **Source:** UCI Machine Learning Repository (ID: 468)
- **Total Samples:** 12,330 sessions
- **Number of Features:** 17 (10 numerical + 7 categorical)
- **Target Variable:** Revenue (binary - 0: no purchase, 1: purchase)
- **Class Distribution:** 10,422 no purchase (84.5%), 1,908 purchase (15.5%)
- **Missing Values:** None

Features include things like number of pages visited (administrative, informational, product related), time spent on each type, bounce rates, exit rates, page values, and some visitor info like browser, OS, whether its a weekend, etc.

The dataset is quite imbalanced - only about 15% of sessions actually end in a purchase. We did 80-20 train-test split with stratification and encoded the categorical columns (Month, VisitorType, Weekend) to numbers.

## c. GitHub Repository Link

> **https://github.com/chndan-2025da04265/ml-assignment2-online-shoppers-classification**

## d. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8779 | 0.8692 | 0.7263 | 0.3403 | 0.4635 | 0.4418 |
| Decision Tree | 0.8463 | 0.7028 | 0.5040 | 0.4948 | 0.4993 | 0.4086 |
| KNN | 0.8727 | 0.7876 | 0.6667 | 0.3560 | 0.4642 | 0.4248 |
| Naive Bayes | 0.8402 | 0.8179 | 0.4852 | 0.5157 | 0.5000 | 0.4053 |
| Random Forest (Ensemble) | 0.8978 | 0.9165 | 0.7181 | 0.5602 | 0.6294 | 0.5771 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression gave strong overall stability with 87.79% accuracy and 0.8692 AUC, which means it ranks sessions reasonably well by purchase probability. However, recall is only 0.3403, so many actual purchase sessions are missed. This happened because the model is conservative on an imbalanced dataset and predicts "no purchase" more often. It is still a useful baseline because precision (0.7263) stays high when it predicts a purchase. |
| Decision Tree | Decision Tree achieved 84.63% accuracy and the lowest AUC (0.7028), indicating weak probability ranking compared to other models. It also showed overfitting behavior: acceptable recall (0.4948) but lower precision (0.5040), meaning it finds more buyers but with many false positives. This model captures non-linear patterns, but a single tree was not robust enough for this dataset. |
| KNN | KNN reached 87.27% accuracy, very close to Logistic Regression, with moderate precision (0.6667) but low recall (0.3560). Since KNN depends heavily on distance, feature scaling was necessary and already applied. Even with tuning, KNN could not separate purchase vs no-purchase sessions strongly in this feature space, so minority-class detection remained weak. |
| Naive Bayes | Naive Bayes produced the lowest accuracy (84.02%) but better recall (0.5157) than LR and KNN, so it captured more actual buyers. Precision is low (0.4852), which means many predicted buyers were false alarms. Its AUC (0.8179) is still reasonable, suggesting probability estimates are useful even if class predictions are noisier. This tradeoff can be acceptable when recall is more important than strict precision. |
| Random Forest (Ensemble) | Random Forest gave the best balanced performance: highest accuracy (0.8978), highest AUC (0.9165), highest F1 (0.6294), highest MCC (0.5771), and highest recall (0.5602). This indicates it not only predicts well overall, but also handles the imbalanced purchase class better than other models. Precision (0.7181) remains strong, so the gain in recall is not achieved by excessive false positives. |
| Overall Winner | **Random Forest** is the best model for this dataset because it performs strongest on almost every metric and gives the best practical balance between identifying buyers (recall) and keeping predictions reliable (precision). Logistic Regression has slightly higher precision (0.7263 vs 0.7181), but it misses far more buyers, so Random Forest is a better final choice for this business use case. |

## e. Live Streamlit App

> **https://ml-assignment2-online-shoppers-classification-2025da04265.streamlit.app/**

## How to Run

```bash
pip install -r requirements.txt
cd model
python train_models.py
cd ..
streamlit run app.py
```

The training script downloads the dataset from UCI, trains all models, and saves everything (pkl files, test_data.csv, metrics.json) so the streamlit app can use them.
