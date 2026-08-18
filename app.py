# ML Assignment 2 - Streamlit App
# Online Shoppers Purchasing Intention - Classification using multiple ML models

import streamlit as st
import pandas as pd
import pickle
import json
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve, average_precision_score)
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

# threshold tuning for class prediction
st.sidebar.write("**Prediction Threshold**")
threshold = st.sidebar.slider("Purchase cutoff probability", 0.10, 0.90, 0.50, 0.01)

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
y_prob = model.predict_proba(X_input)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

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
st.write(f"Using threshold: {threshold:.2f} (Purchase if probability >= {threshold:.2f})")

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
    report = classification_report(y, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    # drop the accuracy row (it shows same value in all columns, looks wrong)
    # show it as a separate metric instead
    overall_acc = report_df.loc['accuracy', 'precision']
    report_df = report_df.drop(index='accuracy')
    report_df['support'] = report_df['support'].astype(int)
    st.write(f"Overall Accuracy: **{overall_acc:.4f}**")
    st.table(report_df[['precision', 'recall', 'f1-score', 'support']])

st.write("---")

# probability-based curves (independent of threshold)
st.subheader("ROC and Precision-Recall Curves")
curve_left, curve_right = st.columns(2)

with curve_left:
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax3.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve')
    ax3.legend(loc='lower right')
    plt.tight_layout()
    st.pyplot(fig3)

with curve_right:
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    ax4.plot(recall_curve, precision_curve, label=f"AP = {pr_auc:.4f}")
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision-Recall Curve')
    ax4.legend(loc='lower left')
    plt.tight_layout()
    st.pyplot(fig4)

st.write("AUC/ROC and PR curves use probabilities, while confusion matrix/report use the selected threshold.")

st.write("---")

# comparison of all models
st.subheader("Comparison of All Models")

# create comparison dataframe from saved metrics
comp_df = pd.DataFrame(saved_metrics).T
st.table(comp_df)
st.write("This table is from training-time metrics saved in metrics.json (default threshold 0.50).")

# bar chart
st.subheader("Performance Chart")
fig2, ax2 = plt.subplots(figsize=(10, 5))
comp_df.plot(kind='bar', ax=ax2, rot=15)
ax2.set_ylabel('Score')
ax2.set_title('Model Comparison')
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax2.set_ylim(0.3, 1.05)
plt.tight_layout()
st.pyplot(fig2)
