import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as SKPCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

st.set_page_config(page_title="PCA Visualizer", layout="wide")
st.title("📊 PCA from Scratch vs Scikit-learn + ML Comparison")

# Dataset
data = load_breast_cancer()
X = data.data
y = data.target

df = pd.DataFrame(X, columns=data.feature_names)
df["target"] = y

# Sidebar
st.sidebar.header("Controls")

n_components = st.sidebar.slider(
    "Number of PCA Components",
    min_value=2,
    max_value=X.shape[1],
    value=2,
)

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "KNN", "Random Forest"]
)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA From Scratch
class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        self.cov_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)

        idx = np.argsort(eigenvalues)[::-1]

        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]

        # (n_components, n_features)
        self.components = self.eigenvectors[:, :self.n_components].T

        self.explained_variance_ratio = (
            self.eigenvalues / np.sum(self.eigenvalues)
        )

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Run PCA
pca_scratch = PCAFromScratch(n_components)

start = time.time()
X_pca_scratch = pca_scratch.fit_transform(X_scaled)
scratch_time = time.time() - start

sk_pca = SKPCA(n_components=n_components)

start = time.time()
X_pca_sklearn = sk_pca.fit_transform(X_scaled)
sk_time = time.time() - start

# Same split before and after PCA
indices = np.arange(len(y))

train_idx, test_idx, y_train, y_test = train_test_split(
    indices,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = X_scaled[train_idx]
X_test = X_scaled[test_idx]

X_train_p = X_pca_sklearn[train_idx]
X_test_p = X_pca_sklearn[test_idx]

def get_model(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=10000)
    elif name == "KNN":
        return KNeighborsClassifier()
    else:
        return RandomForestClassifier(random_state=42)

compression_ratio = (1 - n_components / X.shape[1]) * 100
st.metric("Dimensionality Reduction", f"{compression_ratio:.1f}%")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Overview")
    st.dataframe(df.head())
    st.write("Shape:", X.shape)

with col2:
    st.subheader("Target Distribution")
    st.bar_chart(pd.Series(y).value_counts())

st.subheader("Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pd.DataFrame(X_scaled).corr(), cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("Cumulative Explained Variance")
fig, ax = plt.subplots()
ax.plot(np.cumsum(sk_pca.explained_variance_ratio_), marker="o")
ax.set_xlabel("Components")
ax.set_ylabel("Cumulative Variance")
st.pyplot(fig)

st.subheader("Scree Plot")
fig, ax = plt.subplots()
ax.bar(
    range(1, len(sk_pca.explained_variance_ratio_) + 1),
    sk_pca.explained_variance_ratio_
)
st.pyplot(fig)

if n_components >= 2:
    st.subheader("PCA Projection")
    fig, ax = plt.subplots()
    ax.scatter(
        X_pca_sklearn[:, 0],
        X_pca_sklearn[:, 1],
        c=y,
        cmap="viridis",
        alpha=0.7
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)

st.subheader("Feature Contribution Matrix")

contrib = pd.DataFrame(
    pca_scratch.components.T,
    index=data.feature_names,
    columns=[f"PC{i+1}" for i in range(n_components)]
)

st.dataframe(contrib)

# Before PCA
model = get_model(model_choice)
model.fit(X_train, y_train)
pred = model.predict(X_test)

before_metrics = {
    "Accuracy": accuracy_score(y_test, pred),
    "Precision": precision_score(y_test, pred),
    "Recall": recall_score(y_test, pred),
    "F1": f1_score(y_test, pred),
}

# After PCA
model_p = get_model(model_choice)
model_p.fit(X_train_p, y_train)
pred_p = model_p.predict(X_test_p)

after_metrics = {
    "Accuracy": accuracy_score(y_test, pred_p),
    "Precision": precision_score(y_test, pred_p),
    "Recall": recall_score(y_test, pred_p),
    "F1": f1_score(y_test, pred_p),
}

st.subheader("Metrics Comparison")

comparison = pd.DataFrame(
    {
        "Accuracy": [before_metrics["Accuracy"], after_metrics["Accuracy"]],
        "Precision": [before_metrics["Precision"], after_metrics["Precision"]],
        "Recall": [before_metrics["Recall"], after_metrics["Recall"]],
        "F1": [before_metrics["F1"], after_metrics["F1"]],
    },
    index=["Before PCA", "After PCA"]
)

st.dataframe(comparison)

st.subheader("Confusion Matrix (After PCA)")
cm = confusion_matrix(y_test, pred_p)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

st.subheader("PCA Runtime Comparison")
runtime_df = pd.DataFrame(
    {
        "Method": ["PCA From Scratch", "Scikit-learn PCA"],
        "Time (seconds)": [scratch_time, sk_time],
    }
)
st.dataframe(runtime_df)

st.success(
    "PCA analysis Visualizer"
)
