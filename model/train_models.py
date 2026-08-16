# Training script for ML Assignment 2
# Dataset: Online Shoppers Purchasing Intention (UCI)
# Training 5 classification models and saving them for the streamlit app

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# fetch dataset from UCI repository
from ucimlrepo import fetch_ucirepo
print("Downloading Online Shoppers Purchasing Intention dataset from UCI...")
dataset = fetch_ucirepo(id=468)
X = dataset.data.features
y = dataset.data.targets

# combine into single dataframe for preprocessing
df = X.copy()
df['Revenue'] = y['Revenue'].astype(int)  # True/False -> 1/0

print("Dataset loaded successfully")
print("Shape:", df.shape)
print("Features:", X.shape[1])
print("\nTarget distribution:")
print(df['Revenue'].value_counts())
print(f"\nPositive class (purchased): {df['Revenue'].sum()}")
print(f"Negative class (didn't purchase): {(df['Revenue']==0).sum()}")

# preprocessing - encode categorical columns
# Month -> numeric mapping
month_map = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'June':6,
             'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
df['Month'] = df['Month'].map(month_map)

# VisitorType -> numeric
visitor_map = {'Returning_Visitor': 0, 'New_Visitor': 1, 'Other': 2}
df['VisitorType'] = df['VisitorType'].map(visitor_map)

# Weekend -> int (already boolean)
df['Weekend'] = df['Weekend'].astype(int)

# fill any NaN that appeared from mapping
df = df.fillna(0)

print(f"\nAfter preprocessing shape: {df.shape}")
print("All columns:", list(df.columns))

# separate features and target
X = df.drop('Revenue', axis=1)
y = df['Revenue']

# split into train and test (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# save test data as csv for the streamlit app
test_data = X_test.copy()
test_data['Revenue'] = y_test.values
test_data.to_csv('../test_data.csv', index=False)
print("Test data saved to test_data.csv")

# feature scaling - needed for logistic regression and knn
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# save feature names for the app
feature_names = list(X.columns)
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)

# defining all 5 models
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# these models need scaled data
needs_scaling = ['Logistic Regression', 'KNN']

all_results = {}

print("\n--- Training Models ---\n")

for name, model in models.items():
    # use scaled data for LR and KNN, raw data for others
    if name in needs_scaling:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    # computing all 6 evaluation metrics
    acc = round(accuracy_score(y_test, y_pred), 4)
    auc = round(roc_auc_score(y_test, y_prob), 4)
    prec = round(precision_score(y_test, y_pred), 4)
    rec = round(recall_score(y_test, y_pred), 4)
    f1 = round(f1_score(y_test, y_pred), 4)
    mcc = round(matthews_corrcoef(y_test, y_pred), 4)

    all_results[name] = {
        'Accuracy': acc, 'AUC': auc, 'Precision': prec,
        'Recall': rec, 'F1': f1, 'MCC': mcc
    }

    # save the trained model
    fname = name.lower().replace(' ', '_') + '.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(model, f)

    print(f"{name}:")
    print(f"  Accuracy={acc}, AUC={auc}, Precision={prec}, Recall={rec}, F1={f1}, MCC={mcc}")
    print(f"  -> saved as {fname}")

# save metrics to json so the app can load them
with open('metrics.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\nAll models trained and saved!")
print("Metrics saved to metrics.json")
