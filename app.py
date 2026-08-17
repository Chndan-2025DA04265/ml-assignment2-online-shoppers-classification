# ML Assignment 2 - Streamlit App
# Online Shoppers Purchasing Intention - Classification using multiple ML models

import streamlit as st
import pandas as pd
import pickle
import json
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Assignment 2 - Online Shoppers", layout="wide")

st.title("Online Shoppers Purchasing Intention")
st.write("Predicting whether an online shopping session will end in a purchase or not")
st.write("Dataset: UCI Online Shoppers Purchasing Intention (12,330 sessions, 17 features)")
st.write("---")

# load the saved scaler
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# load saved metrics from training
with open('model/metrics.json', 'r') as f:
    saved_metrics = json.load(f)

# model file paths
model_files = {
    'Logistic Regression': 'model/logistic_regression.pkl',
    'Decision Tree': 'model/decision_tree.pkl',
    'KNN': 'model/knn.pkl',
    'Naive Bayes': 'model/naive_bayes.pkl',
    'Random Forest': 'model/random_forest.pkl'
}

# models that need scaled features
needs_scaling = ['Logistic Regression', 'KNN']

# target class names
target_names = ['No Purchase', 'Purchase']

# --- Sidebar ---
st.sidebar.header("Settings")

# data upload option
st.sidebar.write("**Data Input**")
data_choice = st.sidebar.radio("Choose data:", ["Use default test data", "Upload CSV"])

if data_choice == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload test CSV file", type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        st.warning("Please upload a CSV file to continue")
        st.stop()
else:
    df = pd.read_csv('test_data.csv')

# check if target column exists
if 'Revenue' not in df.columns:
    st.error("Error: CSV file must have a 'Revenue' column (0 or 1)")
    st.stop()

# model selection dropdown
st.sidebar.write("**Select Model**")
chosen_model = st.sidebar.selectbox("Model:", list(model_files.keys()))

# show dataset info
st.subheader("Dataset Preview")
st.dataframe(df.head(10))
st.write(f"Total samples: {len(df)} | Features: {df.shape[1]-1} | Classes: {df['Revenue'].nunique()}")

# separate features and target
X = df.drop('Revenue', axis=1)
y = df['Revenue']

# load the selected model
with open(model_files[chosen_model], 'rb') as f:
    model = pickle.load(f)

# apply scaling if needed
if chosen_model in needs_scaling:
    X_input = scaler.transform(X)
else:
    X_input = X.values

# make predictions
y_pred = model.predict(X_input)
y_prob = model.predict_proba(X_input)[:, 1]

# calculate evaluation metrics
acc = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_prob)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
mcc = matthews_corrcoef(y, y_pred)

# display metrics
st.write("---")
st.subheader(f"Results for {chosen_model}")

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)
col1.metric("Accuracy", f"{acc:.4f}")
col2.metric("AUC Score", f"{auc:.4f}")
col3.metric("Precision", f"{prec:.4f}")
col4.metric("Recall", f"{rec:.4f}")
col5.metric("F1 Score", f"{f1:.4f}")
col6.metric("MCC", f"{mcc:.4f}")

st.write("---")

# confusion matrix
left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(chosen_model)
    plt.tight_layout()
    st.pyplot(fig)

with right:
    st.subheader("Classification Report")
    report = classification_report(y, y_pred, target_names=target_names,output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.text(report_df)

st.write("---")

# comparison of all models
st.subheader("Comparison of All Models")

# create comparison dataframe from saved metrics
comp_df = pd.DataFrame(saved_metrics).T
st.table(comp_df)

# bar chart
st.subheader("Performance Chart")
fig2, ax2 = plt.subplots(figsize=(10, 5))
comp_df.plot(kind='bar', ax=ax2, rot=15)
ax2.set_ylabel('Score')
ax2.set_title('Model Comparison')
ax2.legend(loc='lower right')
ax2.set_ylim(0.3, 1.05)
plt.tight_layout()
st.pyplot(fig2)
