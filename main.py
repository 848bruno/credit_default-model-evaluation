import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             accuracy_score, precision_recall_curve, auc)
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_samples = 1000
data = {
    'Loan_ID': [f'LP{str(i).zfill(6)}' for i in range(num_samples)],
    'Gender': np.random.choice(['MALE', 'FEMALE'], num_samples),
    'Dependents': np.random.randint(0, 4, num_samples),
    'ApplicantIncome': np.random.normal(5000, 1500, num_samples).astype(int),
    'CoapplicantIncome': np.random.exponential(1000, num_samples).astype(int),
    'LoanAmount': np.random.normal(150, 50, num_samples).astype(int),
    'Loan_Amount_Term': np.random.choice([360, 180, 120, 240], num_samples),
    'Credit_History': np.random.choice([0, 1], num_samples, p=[0.3, 0.7])
}

# Create target variable
log_odds = (-0.5 * (data['LoanAmount']/1000) 
            + 0.3 * (data['ApplicantIncome']/1000)
            - 0.2 * data['Dependents']
            + 1.5 * data['Credit_History']
            + np.random.normal(0, 0.5, num_samples))

prob_default = 1 / (1 + np.exp(-log_odds))
data['Default'] = np.random.binomial(1, prob_default)

df = pd.DataFrame(data)

# Preprocessing
X = df.drop(['Loan_ID', 'Default'], axis=1)
y = df['Default']
X = pd.get_dummies(X, columns=['Gender', 'Loan_Amount_Term'], drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Initialize and train models
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Generate predictions and probabilities
y_pred_lr = log_reg.predict(X_test)
y_prob_lr = log_reg.predict_proba(X_test)[:, 1]

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Calculate metrics function
def calculate_metrics(name, y_true, y_pred, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {
        'Model': name,
        'Precision': f"{precision_score(y_true, y_pred):.2f}",
        'Recall': f"{recall_score(y_true, y_pred):.2f}",
        'F1-Score': f"{f1_score(y_true, y_pred):.2f}",
        'Accuracy': f"{accuracy_score(y_true, y_pred):.2f}",
        'PR AUC': f"{auc(recall, precision):.2f}"
    }

# Create metrics dataframe
metrics = [
    calculate_metrics('Logistic Regression', y_test, y_pred_lr, y_prob_lr),
    calculate_metrics('Random Forest', y_test, y_pred_rf, y_prob_rf)
]

results_df = pd.DataFrame(metrics)

# Create figure with 3 subplots
plt.figure(figsize=(15, 10))

# PR Curves
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((2, 2), (1, 0))
ax3 = plt.subplot2grid((2, 2), (1, 1))

# Metrics Table
cell_text = results_df.values.tolist()
columns = results_df.columns.tolist()

ax1.axis('off')
table = ax1.table(cellText=cell_text,
                 colLabels=columns,
                 loc='center',
                 cellLoc='center',
                 colColours=['#f0f0f0']*len(columns))
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)
ax1.set_title('Model Evaluation Metrics', fontsize=14, pad=20)

# Random Forest PR Curve
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_prob_rf)
ax2.plot(recall_rf, precision_rf, color='darkorange', lw=2)
ax2.plot([0, 1], [1, 0], linestyle='--', color='navy')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Random Forest PR Curve')
ax2.grid(True)

# Logistic Regression PR Curve
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_prob_lr)
ax3.plot(recall_lr, precision_lr, color='darkgreen', lw=2)
ax3.plot([0, 1], [1, 0], linestyle='--', color='navy')
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('Logistic Regression PR Curve')
ax3.grid(True)

plt.tight_layout()
plt.show()