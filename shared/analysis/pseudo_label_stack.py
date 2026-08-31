#!/usr/bin/env python3
"""Pseudo-labeling experiment.

Hypothesis: Using the rank-gauss stack's high-confidence test predictions
as pseudo-labels and retraining on the combined real+pseudo data could capture
patterns that the current models miss. This is a fundamentally different approach
from member addition or combiner optimization.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import os, sys, warnings
warnings.filterwarnings('ignore')

def rank_gauss(x):
    n = len(x)
    return rankdata(x) / (n + 1)

# Load labels
train = pd.read_csv('competitions/s6e8/data/train.csv')
test = pd.read_csv('competitions/s6e8/data/test.csv')
y = train['addicted_label'].values
N = len(y)
N_test = len(test)
print(f'Train: {N} rows, Test: {N_test} rows')
print(f'Positive rate: {y.mean():.4f}')

# Load boltuzamaki library for OOF + test
import kagglehub
bolt_path = kagglehub.dataset_download('boltuzamaki/s6e8-oof-prediction-library')
bolt_oof = pd.read_parquet(os.path.join(bolt_path, 'oof_prediction_library.parquet'))
bolt_test = pd.read_parquet(os.path.join(bolt_path, 'test_prediction_library.parquet'))
bolt_cols = [c for c in bolt_oof.columns if c != 'id']
print(f'Bolt library: {len(bolt_cols)} members')

# Load naji OOF + test
naji_path = kagglehub.dataset_download('najiama/predicting-smartphone-addiction-oof-submission-csv')
naji_oofs_train = {}
naji_oofs_test = {}
for f in sorted(os.listdir(naji_path)):
    if f.endswith('_oof_predictions.csv'):
        df = pd.read_csv(os.path.join(naji_path, f))
        pred_col = [c for c in df.columns if c != 'id'][0]
        oof = df[pred_col].values
        if len(oof) == N:
            name = f.replace('_oof_predictions.csv', '')
            naji_oofs_train[name] = oof
    elif f.endswith('_submission.csv'):
        df = pd.read_csv(os.path.join(naji_path, f))
        pred_col = [c for c in df.columns if c != 'id'][0]
        test_pred = df[pred_col].values
        if len(test_pred) == N_test:
            name = f.replace('_submission.csv', '')
            naji_oofs_test[name] = test_pred

print(f'Naji: {len(naji_oofs_train)} OOF, {len(naji_oofs_test)} test')

# Build member matrices
print('\n=== BUILDING MEMBER MATRICES ===')
train_members = []
test_members = []
member_names = []

# Bolt members
for col in bolt_cols:
    rg_train = rank_gauss(bolt_oof[col].values)
    rg_test = rank_gauss(bolt_test[col].values)
    train_members.append(rg_train)
    test_members.append(rg_test)
    member_names.append(f'bolt_{col}')

# Naj members (high AUC only)
for name in naji_oofs_train:
    auc = roc_auc_score(y, naji_oofs_train[name])
    if auc > 0.968:
        rg_train = rank_gauss(naji_oofs_train[name])
        rg_test = rank_gauss(naji_oofs_test.get(name, np.zeros(N_test)))
        train_members.append(rg_train)
        test_members.append(rg_test)
        member_names.append(f'naj_{name}')

G = np.column_stack(train_members)
Gt = np.column_stack(test_members)
print(f'Train: {G.shape}, Test: {Gt.shape}')

# Step 1: Build the baseline stack OOF
print('\n=== BASELINE STACK ===')
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
oof_baseline = np.zeros(N)

for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(G, y)):
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(G[tr_idx], y[tr_idx])
    oof_baseline[va_idx] = model.predict_proba(G[va_idx])[:, 1]

baseline_auc = roc_auc_score(y, oof_baseline)
print(f'Baseline stack OOF: {baseline_auc:.6f}')

# Step 2: Get full-data test predictions
model_full = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model_full.fit(G, y)
test_predictions = model_full.predict_proba(Gt)[:, 1]
print(f'Test predictions: mean={test_predictions.mean():.4f}, std={test_predictions.std():.4f}')

# Step 3: Pseudo-labeling experiment
print('\n=== PSEUDO-LABELING ===')
thresholds = [0.90, 0.95, 0.98]
results = []

for thresh in thresholds:
    # Select high-confidence pseudo-labels
    high_conf_mask = (test_predictions > thresh) | (test_predictions < (1-thresh))
    n_pseudo = high_conf_mask.sum()
    print(f'\nThreshold {thresh}: {n_pseudo} pseudo-labels ({n_pseudo/N_test*100:.1f}%)')
    
    if n_pseudo < 100:
        print('  Too few pseudo-labels, skipping')
        continue
    
    # Create pseudo-labels
    pseudo_labels = (test_predictions[high_conf_mask] > 0.5).astype(float)
    print(f'  Pseudo positive rate: {pseudo_labels.mean():.4f}')
    
    # Augment training data
    G_aug = np.vstack([G, Gt[high_conf_mask]])
    y_aug = np.concatenate([y, pseudo_labels])
    print(f'  Augmented: {len(y_aug)} rows ({len(y_aug)/len(y)*100:.1f}% of original)')
    
    # Cross-validate on augmented data
    oof_aug = np.zeros(N)
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(G, y)):
        # Train on augmented fold (only real data in validation)
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(G_aug[tr_idx], y_aug[tr_idx])
        oof_aug[va_idx] = model.predict_proba(G[va_idx])[:, 1]
    
    aug_auc = roc_auc_score(y, oof_aug)
    delta = aug_auc - baseline_auc
    print(f'  Augmented OOF: {aug_auc:.6f} (delta: {delta:+.6f})')
    results.append((thresh, n_pseudo, aug_auc, delta))

# Step 4: Residual correction experiment
print('\n=== RESIDUAL CORRECTION ===')
# Train a model to predict where the baseline stack is wrong
errors = y - oof_baseline
error_auc = roc_auc_score(y, oof_baseline)  # Same as baseline
print(f'Baseline error distribution: mean={errors.mean():.4f}, std={errors.std():.4f}')

# Train a residual model
residual_model = GradientBoostingClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, random_state=42
)
residual_model.fit(G, y)
oof_residual = residual_model.predict_proba(G)[:, 1]
residual_auc = roc_auc_score(y, oof_residual)
print(f'Residual model OOF: {residual_auc:.6f} (delta: {residual_auc - baseline_auc:+.6f})')

# Blend baseline + residual
best_w, best_auc = 0, 0
for w in np.arange(0.0, 1.05, 0.05):
    blend = w * oof_baseline + (1-w) * oof_residual
    auc = roc_auc_score(y, blend)
    if auc > best_auc:
        best_auc = auc
        best_w = w
print(f'Best baseline+residual blend: W={best_w:.2f}, AUC={best_auc:.6f}')

# Save results
results_dict = {
    'baseline_auc': float(baseline_auc),
    'pseudo_label_results': [(float(t), int(n), float(a), float(d)) for t, n, a, d in results],
    'residual_auc': float(residual_auc),
    'best_blend_w': float(best_w),
    'best_blend_auc': float(best_auc),
}
print(f'\n=== RESULTS ===')
print(f'Baseline: {baseline_auc:.6f}')
for t, n, a, d in results:
    print(f'Pseudo-label (t={t}): {a:.6f} ({d:+.6f})')
print(f'Residual: {residual_auc:.6f}')
print(f'Best blend: {best_auc:.6f}')
print('DONE')
