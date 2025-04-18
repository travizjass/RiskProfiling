import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import shap
import json

# Load your data
df = pd.read_csv("csvs/filtered.csv")  # Replace with your actual file

# Clean ZIP-related features
df['averageIncome'] = df['averageIncome'].replace(r'[\$,]', '', regex=True).astype(float)
df['populationDensity'] = df['populationDensity'].replace(r'[^0-9.]', '', regex=True).astype(float)
df['educationPercentage'] = df['educationPercentage'].str.replace('%', '').astype(float)
df['employmentPercentage'] = df['employmentPercentage'].str.replace('%', '').astype(float)
df['publicInsurancePercentage'] = df['publicInsurancePercentage'].str.replace('%', '').astype(float)
df['noInsurancePercentage'] = df['noInsurancePercentage'].str.replace('%', '').astype(float)

# Drop non-feature columns
df_base = df.drop(columns=['id', 'patient_id', 'icd_code', 'zip_code'])

# Fill missing values
df_base = df_base.fillna(df_base.median())

# Define healthy ranges for clinical & demographic features (adjusted)
healthy_ranges = {
    # Clinical vitals
    'bp_systolic': (90, 140),
    'bp_diastolic': (60, 80),
    'pulse': (60, 100),
    'respiration': (12, 20),
    'temperature': (97.0, 99.0),  # in Fahrenheit
    'oxygen_saturation': (95, 100),
    'bmi': (18.5, 24.9),
    # Severity and demographics
    'severity_score': (0, 6),
    'age': (18, 65),
    # Socioeconomic (ZIP-level)
    'averageIncome': (30000, 1000000),
    'populationDensity': (100, 10000),
    'educationPercentage': (20, 100),
    'employmentPercentage': (50, 100),
    'publicInsurancePercentage': (0, 50),
    'noInsurancePercentage': (0, 10)
}

# Engineer deviation features for all metrics
df_dev = df_base.copy()
for feat, (low, high) in healthy_ranges.items():
    df_dev[f"{feat}_below"] = np.where(df_dev[feat] < low, low - df_dev[feat], 0)
    df_dev[f"{feat}_above"] = np.where(df_dev[feat] > high, df_dev[feat] - high, 0)

# Split into health and socio feature sets
health_feats = ['bp_systolic', 'bp_diastolic', 'pulse', 'respiration',
                'temperature', 'oxygen_saturation', 'bmi', 'severity_score', 'age']
socio_feats = ['averageIncome', 'populationDensity', 'educationPercentage',
               'employmentPercentage', 'publicInsurancePercentage', 'noInsurancePercentage']

# include their deviation counterparts
health_devs = [c for c in df_dev.columns if any(c.startswith(h + "_") for h in health_feats)]
socio_devs = [c for c in df_dev.columns if any(c.startswith(s + "_") for s in socio_feats)]

# Create separate dataframes
df_health = df_base[health_feats].join(df_dev[health_devs])
df_socio = df_base[socio_feats].join(df_dev[socio_devs])

# Function to compute risk via IsolationForest + GMM
def compute_risks(X):
    iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    iso.fit(X)
    iso_score = -iso.decision_function(X)
    iso_risk = MinMaxScaler((0, 100)).fit_transform(iso_score.reshape(-1, 1)).flatten()

    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)
    probs = gmm.predict_proba(X)
    cluster_sizes = np.bincount(gmm.predict(X))
    high_cluster = np.argmin(cluster_sizes)
    gmm_risk = probs[:, high_cluster] * 100

    return 0.5 * iso_risk + 0.5 * gmm_risk, iso, gmm

# Compute separate risks
risk_health, iso_h, gmm_h = compute_risks(df_health)
risk_socio, iso_s, gmm_s = compute_risks(df_socio)

# Combine into final score: 80% health + 20% socioeconomic
df['risk_health'] = risk_health
df['risk_socio'] = risk_socio
df['risk_combined'] = 0.8 * risk_health + 0.2 * risk_socio

# Save scored patients
df.to_csv("scored_patients.csv", index=False)

# Visualize distribution of combined risk
plt.figure()
plt.hist(df['risk_combined'], bins=50, edgecolor='black')
plt.title('Distribution of Combined Risk Scores')
plt.xlabel('Risk Score (0-100)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("images/risk_score_distribution.png")

# Scatter plot of all patients by risk
plt.figure(figsize=(10, 6))
plt.scatter(range(len(df)), df['risk_combined'], alpha=0.6)
plt.title('Scatter Plot of Patient Risk Scores')
plt.xlabel('Patient Index')
plt.ylabel('Risk Score (0-100)')
plt.grid(True)
plt.tight_layout()
plt.savefig("images/risk_score_scatter.png")

# SHAP explainability: combine scaled health (×0.8) and socio (×0.2) for global importance
X_explain = pd.concat([df_health * 0.8, df_socio * 0.2], axis=1)
explainer = shap.Explainer(iso_h, X_explain)
shap_vals = explainer(X_explain)

# Global bar plot
plt.figure()
shap.plots.bar(shap_vals, show=False)
plt.title('Global Feature Impact (80% Health, 20% Socio)')
plt.tight_layout()
plt.savefig('images/shap_global_bar.png')

# Pie chart of top impacts
top_n = 15
mean_shap = np.abs(shap_vals.values).mean(axis=0)
feat_imp = pd.Series(mean_shap, index=X_explain.columns)
top_feats = feat_imp.nlargest(top_n)
others = feat_imp.drop(top_feats.index).sum()
pie_data = pd.concat([top_feats, pd.Series({'Other': others})])
plt.figure(figsize=(8, 8))
plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
plt.title('Feature Impact Proportions (80% Health, 20% Socio)')
plt.tight_layout()
plt.savefig('images/shap_pie.png')

# Example waterfall for first patient
patient_idx = 0
plt.figure()
shap.plots.waterfall(shap_vals[patient_idx], show=False)
plt.tight_layout()
plt.savefig(f'images/shap_waterfall_{patient_idx}.png')

# Create JSON of patient explanations
explain_json = []
for i, row in df.iterrows():
    patient_shap = shap_vals[i]
    sorted_feats = sorted(zip(X_explain.columns, patient_shap.values), key=lambda x: abs(x[1]), reverse=True)[:10]
    total = sum(abs(val) for _, val in sorted_feats)
    features = {k: f"{(abs(v)/total)*100:.2f}%" for k, v in sorted_feats}
    explain_json.append({
        "id": row['id'],
        "risk": f"{row['risk_combined']:.2f}",
        "features": features
    })

# Sort JSON by risk in descending order
explain_json = sorted(explain_json, key=lambda x: float(x['risk']), reverse=True)

with open("patient_risk_explanations.json", "w") as f:
    json.dump(explain_json, f, indent=2)

# Filter patients with risk > 70%
high_risk_patients = df[df['risk_combined'] > 70].index.tolist()
shap_high_risk = shap_vals[high_risk_patients]

# Mean absolute SHAP values across high-risk patients
mean_high_risk_shap = np.abs(shap_high_risk.values).mean(axis=0)
high_risk_feat_imp = pd.Series(mean_high_risk_shap, index=X_explain.columns).sort_values(ascending=False)

# Plot bar chart
plt.figure(figsize=(10, 6))
high_risk_feat_imp.head(15).plot(kind='bar', color='tomato', edgecolor='black')
plt.title('Top Contributing Features for High-Risk Patients (>70%)')
plt.ylabel('Mean Absolute SHAP Value')
plt.xlabel('Feature')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('images/shap_high_risk_bar.png')

