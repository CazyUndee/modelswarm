#!/usr/bin/env python3
"""Stack + naji blend experiment.

Hypothesis: naji 19_blend (AUC 0.970099) is a genuinely different model that
could boost the rank-gauss stack when blended properly.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import os, sys, warnings, json
warnings.filterwarnings('ignore')

def rank_gauss(x):
    n = len(x)
    return rankdata(x) / (n + 1)

# Load labels
train = pd.read_csv('competitions/s6e8/data/train.csv')
y = train['addicted_label'].values
N = len(y)
print(f'Labels: {N} rows, positive rate={y.mean():.4f}')

# Load existing boltuzamaki library
print('\n=== LOADING BOLTUZAMAKI LIBRARY ===')
import kagglehub
bolt_path = kagglehub.dataset_download('boltuzamaki/s6e8-oof-prediction-library')
bolt_files = [f for f in os.listdir(bolt_path) if f.endswith('.parquet')]
print(f'Bolt files: {bolt_files}')
bolt_df = pd.read_parquet(os.path.join(bolt_path, [f for f in bolt_files if 'oof' in f.lower()][0]))
bolt_cols = [c for c in bolt_df.columns if c != 'id']
print(f'Bolt library: {len(bolt_cols)} members, {len(bolt_df)} rows')

# Load naji OOF
print('\n=== LOADING NAJI OOF ===')
naji_path = kagglehub.dataset_download('najiama/predicting-smartphone-addiction-oof-submission-csv')
naji_oofs = {}
for f in sorted(os.listdir(naji_path)):
    if f.endswith('_oof_predictions.csv'):
        df = pd.read_csv(os.path.join(naji_path, f))
        pred_col = [c for c in df.columns if c != 'id'][0]
        oof = df[pred_col].values
        if len(oof) == N:
            auc = roc_auc_score(y, oof)
            name = f.replace('_oof_predictions.csv', '')
            naji_oofs[name] = oof
            print(f'  Naj {name}: AUC={auc:.6f}')

# Load FM members
print('\n=== LOADING FM MEMBERS ===')
fm_path = kagglehub.dataset_download('raykkretzschmar/s6e8-fm-lattice-blend-members')
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

fm_oofs = {}
for name in ['fmdeep', 'fmnum', 'fmplr', 'fmpure', 'fmwide']:
    path = os.path.join(fm_path, f'oof_{name}.npy')
    if os.path.exists(path):
        raw = np.load(path)
        prob = sigmoid(raw)
        auc = roc_auc_score(y, prob)
        fm_oofs[name] = prob
        print(f'  FM {name}: AUC={auc:.6f}')

# Load weak50 + szymonkapiski for completeness
print('\n=== LOADING SZYMONKAPISKI LIBRARY ===')
try:
    sz_path = kagglehub.dataset_download('szymonkapiski/s6e8-oof-library-47-models')
    sz_files = [f for f in os.listdir(sz_path) if f.endswith('.npy') and f.startswith('oof_')]
    sz_oofs = {}
    for f in sz_files:
        path = os.path.join(sz_path, f)
        oof = np.load(path)
        if len(oof) == N:
            name = f.replace('oof_', '').replace('.npy', '')
            auc = roc_auc_score(y, oof)
            sz_oofs[name] = oof
            print(f'  SZ {name}: AUC={auc:.6f}')
except Exception as e:
    print(f'  SZ load failed: {e}')
    sz_oofs = {}

# Build member matrix
print('\n=== BUILDING MEMBER MATRIX ===')
all_members = {}

# Bolt members (rank-gauss)
for col in bolt_cols:
    rg = rank_gauss(bolt_df[col].values)
    all_members[f'bolt_{col}'] = rg

# SZ members (already OOF probabilities, rank-gauss)
for name, oof in sz_oofs.items():
    rg = rank_gauss(oof)
    all_members[f'sz_{name}'] = rg

# Naj members (already probabilities, rank-gauss)
for name, oof in naji_oofs.items():
    rg = rank_gauss(oof)
    all_members[f'naj_{name}'] = rg

# FM members (already probabilities after sigmoid, rank-gauss)
for name, oof in fm_oofs.items():
    rg = rank_gauss(oof)
    all_members[f'fm_{name}'] = rg

member_names = list(all_members.keys())
G = np.column_stack([all_members[n] for n in member_names])
print(f'Total members: {G.shape[1]}, rows: {G.shape[0]}')

# Deduplicate: check for near-duplicate columns (>0.999 correlation)
print('\n=== DEDUPLICATION ===')
keep = list(range(G.shape[1]))
for i in range(G.shape[1]):
    if i not in keep:
        continue
    for j in range(i+1, G.shape[1]):
        if j not in keep:
            continue
        corr = np.corrcoef(G[:, i], G[:, j])[0, 1]
        if abs(corr) > 0.999:
            print(f'  DROP {member_names[j]} (corr={corr:.4f} with {member_names[i]})')
            keep.remove(j)

G = G[:, keep]
kept_names = [member_names[i] for i in keep]
print(f'After dedup: {G.shape[1]} members')

# Cross-validation
print('\n=== CROSS-VALIDATION ===')
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
oof_stack = np.zeros(N)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(G, y)):
    print(f'Fold {fold_idx+1}/{n_folds}...', end=' ', flush=True)
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(G[train_idx], y[train_idx])
    oof_stack[val_idx] = model.predict_proba(G[val_idx])[:, 1]
    auc = roc_auc_score(y[val_idx], oof_stack[val_idx])
    print(f'AUC={auc:.6f}')

stack_auc = roc_auc_score(y, oof_stack)
print(f'\nStack OOF AUC: {stack_auc:.6f}')

# Load vault base for blending
print('\n=== LOADING VAULT BASE ===')
try:
    vault_path = kagglehub.dataset_download('anthonytherrien/predicting-smartphone-addiction-vault')
    vault_files = [f for f in os.listdir(vault_path) if f.endswith('.csv')]
    if vault_files:
        vault_df = pd.read_csv(os.path.join(vault_path, vault_files[0]))
        vault_pred = vault_df['addicted_label'].values
        if len(vault_pred) >= N:
            vault_pred = vault_pred[:N]
            vault_auc = roc_auc_score(y, vault_pred)
            print(f'Vault: AUC={vault_auc:.6f}, shape={vault_pred.shape}')
        else:
            print(f'Vault too short: {len(vault_pred)} < {N}')
            vault_pred = None
    else:
        print('No vault CSV found')
        vault_pred = None
except Exception as e:
    print(f'Vault load failed: {e}')
    vault_pred = None

# Test blend weights
if vault_pred is not None:
    print('\n=== BLEND WEIGHTS ===')
    best_w, best_auc = 0, 0
    for w in np.arange(0.0, 1.05, 0.05):
        blend = w * oof_stack + (1-w) * vault_pred
        auc = roc_auc_score(y, blend)
        marker = ' <--' if auc > best_auc else ''
        print(f'W={w:.2f}: AUC={auc:.6f}{marker}')
        if auc > best_auc:
            best_auc = auc
            best_w = w
    print(f'\nBest: W={best_w:.2f}, AUC={best_auc:.6f}')
    
    # Fine-tune around best
    print('\n=== FINE-TUNE ===')
    fine_best_w, fine_best_auc = best_w, best_auc
    for w in np.arange(max(0, best_w-0.1), min(1, best_w+0.1), 0.01):
        blend = w * oof_stack + (1-w) * vault_pred
        auc = roc_auc_score(y, blend)
        if auc > fine_best_auc:
            fine_best_auc = auc
            fine_best_w = w
    print(f'Fine-tuned: W={fine_best_w:.2f}, AUC={fine_best_auc:.6f}')
else:
    fine_best_w = 1.0
    fine_best_auc = stack_auc

# Save OOF
np.save('shared/analysis/oof_stack_naji.npy', oof_stack)

# Generate submission
print('\n=== GENERATING SUBMISSION ===')
# Load test data
test = pd.read_csv('competitions/s6e8/data/test.csv')
test_ids = test['id'].values

# Build test stack
print('Building test stack...')
# Load bolt test
bolt_test = pd.read_parquet(os.path.join(bolt_path, [f for f in bolt_files if 'test' in f.lower()][0]))
test_members = []
for col in bolt_cols:
    if col in bolt_test.columns:
        rg = rank_gauss(bolt_test[col].values)
        test_members.append(rg)

# Load naji test
for f in sorted(os.listdir(naji_path)):
    if f.endswith('_submission.csv'):
        df = pd.read_csv(os.path.join(naji_path, f))
        pred_col = [c for c in df.columns if c != 'id'][0]
        test_pred = df[pred_col].values
        if len(test_pred) == len(test_ids):
            name = f.replace('_submission.csv', '')
            if f'naj_{name}' in [kept_names[i] for i in range(len(kept_names))]:
                rg = rank_gauss(test_pred)
                test_members.append(rg)

# Load FM test
for name in ['fmdeep', 'fmnum', 'fmplr', 'fmpure', 'fmwide']:
    path = os.path.join(fm_path, f'test_{name}.npy')
    if os.path.exists(path):
        raw = np.load(path)
        prob = sigmoid(raw)
        if f'fm_{name}' in kept_names:
            rg = rank_gauss(prob)
            test_members.append(rg)

Gt = np.column_stack(test_members)
print(f'Test stack: {Gt.shape}')

# Fit full model for test predictions
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(G, y)
test_stack = model.predict_proba(Gt)[:, 1]

# Blend with vault
if vault_pred is not None:
    vault_test = pd.read_csv(os.path.join(vault_path, 'submission.csv'))
    vault_test_pred = vault_test['addicted_label'].values[:len(test_ids)]
    test_blend = fine_best_w * test_stack + (1-fine_best_w) * vault_test_pred
else:
    test_blend = test_stack

# Save submission
sub_df = pd.DataFrame({'id': test_ids, 'addicted_label': test_blend})
sub_path = f'shared/analysis/submission_stack_naji_w{fine_best_w:.2f}.csv'
sub_df.to_csv(sub_path, index=False)
print(f'Saved submission: {sub_path}')

# Summary
print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f'Total members: {G.shape[1]}')
print(f'Stack OOF: {stack_auc:.6f}')
if vault_pred is not None:
    print(f'Vault OOF: {roc_auc_score(y, vault_pred):.6f}')
    print(f'Best blend W: {fine_best_w:.2f}')
    print(f'Best blend OOF: {fine_best_auc:.6f}')
print(f'Submission: {sub_path}')
print('DONE')
