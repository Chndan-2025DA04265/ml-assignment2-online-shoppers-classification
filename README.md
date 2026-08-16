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

> **https://github.com/chndan-2025da04265/ml-assignment2-online-shoppers**

## d. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | - | - | - | - | - | - |
| Decision Tree | - | - | - | - | - | - |
| KNN | - | - | - | - | - | - |
| Naive Bayes | - | - | - | - | - | - |
| Random Forest (Ensemble) | - | - | - | - | - | - |

> **Note:** Run `python model/train_models.py` to get actual values. The metrics will be printed and saved to `model/metrics.json`. Update this table with those values.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Decent baseline model. Scaling was important for convergence. Struggles a bit with the imbalanced classes since it tries to find a linear boundary. |
| Decision Tree | Overfits easily on this data. Got high training accuracy but test accuracy drops. Recall on purchase class was low because of imbalance. |
| KNN | Needed scaling since features have very different ranges. k=7 gave okay results. Slower to predict compared to other models because of distance calculations. |
| Naive Bayes | Independence assumption doesn't really hold here (page visits and duration are obviously correlated) so accuracy is a bit lower. But AUC is decent. |
| Random Forest (Ensemble) | Best model overall. Combining many trees helps with the overfitting issue we saw in single Decision Tree. Also handles imbalance better. |
| Overall Winner | _(fill after running - check metrics.json)_ |

## e. Live Streamlit App

> **https://<YOUR_APP>.streamlit.app**
>
> _(update after deploying)_

## How to Run

```bash
pip install -r requirements.txt
cd model
python train_models.py
cd ..
streamlit run app.py
```

The training script downloads the dataset from UCI, trains all models, and saves everything (pkl files, test_data.csv, metrics.json) so the streamlit app can use them.
