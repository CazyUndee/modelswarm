#!/usr/bin/env python3
"""
EXP-007: LightGBM 5-fold CV with champion feature set + improvements.
Goal: Beat EXP-006 champion OOF 0.95965 on playground-series-s6e8.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('C:/Users/roone.DESKTOP-QK3UG2M/Downloads/projects/ModelSwarm/competitions/s6e8/data/train.csv')
test = pd.read_csv('C:/Users/roone.DESKTOP-QK3UG2M/Downloads/projects/ModelSwarm/competitions/s6e8/data/test.csv')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Feature engineering (same as EXP-006 champion)
def engineer_features(df):
    df = df.copy()
    # Ratios
    df['social_ratio'] = df['social_media_hours'] / (df['daily_screen_time_hours'] + 1e-6)
    df['gaming_ratio'] = df['gaming_hours'] / (df['daily_screen_time_hours'] + 1e-6)
    df['weekend_boost'] = df['weekend_screen_time'] / (df['daily_screen_time_hours'] + 1e-6)
    df['sleep_debt'] = df['sleep_hours'] - df['daily_screen_time_hours']
    df['total_leisure'] = df['social_media_hours'] + df['gaming_hours']
    df['screen_x_social'] = df['daily_screen_time_hours'] * df['social_media_hours']
    df['weekend_ratio'] = df['weekend_screen_time'] / (df['total_leisure'] + 1e-6)
    df['notif_per_hour'] = df['notifications_per_day'] / (df['daily_screen_time_hours'] + 1e-6)
    df['app_per_notif'] = df['app_opens_per_day'] / (df['notifications_per_day'] + 1e-6)
    return df

train = engineer_features(train)
test = engineer_features(test)

# Features to use
numeric_features = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
                    'work_study_hours', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day',
                    'weekend_screen_time', 'social_ratio', 'gaming_ratio', 'weekend_boost',
                    'sleep_debt', 'total_leisure', 'screen_x_social', 'weekend_ratio',
                    'notif_per_hour', 'app_per_notif']

cat_features = ['gender', 'stress_level', 'academic_work_impact']
all_features = numeric_features + cat_features

X = train[all_features].copy()
y = train['addicted_label'].values
X_test = test[all_features].copy()

# Convert categorical features to 'category' dtype for LightGBM native handling
for col in cat_features:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

print(f"Features: {len(all_features)} ({len(numeric_features)} numeric + {len(cat_features)} categorical)")

# LightGBM parameters (tuned)
lgb_params = {
    'n_estimators': 1200,
    'learning_rate': 0.03,
    'num_leaves': 63,
    'max_depth': -1,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_jobs': -1,
    'verbose': -1,
    'random_state': 42,
}

# 5-fold Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1}/5 ---")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model = LGBMClassifier(**lgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[]
    )
    
    val_preds = model.predict_proba(X_val)[:, 1]
    fold_auc = roc_auc_score(y_val, val_preds)
    fold_scores.append(fold_auc)
    oof_preds[val_idx] = val_preds
    
    # Test predictions
    test_preds += model.predict_proba(X_test)[:, 1] / 5
    
    print(f"Fold {fold+1} AUC: {fold_auc:.6f}")

# Overall OOF
oof_auc = roc_auc_score(y, oof_preds)
print(f"\n{'='*50}")
print(f"OOF AUC: {oof_auc:.6f}")
print(f"Fold scores: {[f'{s:.6f}' for s in fold_scores]}")
print(f"Mean fold: {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")
print(f"Champion (EXP-006): 0.95965")
print(f"Delta: {oof_auc - 0.95965:+.6f}")
print(f"{'='*50}")

# Save results
results = {
    'oof_auc': oof_auc,
    'fold_scores': fold_scores,
    'mean_fold': np.mean(fold_scores),
    'std_fold': np.std(fold_scores),
    'delta_vs_champion': oof_auc - 0.95965,
    'features': all_features,
    'params': lgb_params,
}

# Save submission
submission = pd.DataFrame({'id': test['id'], 'addiction': test_preds})
submission.to_csv('C:/Users/roone.DESKTOP-QK3UG2M/Downloads/projects/ModelSwarm/competitions/s6e8/data/submission_exp007.csv', index=False)
print(f"\nSubmission saved: submission_exp007.csv")

# Save OOF predictions
oof_df = pd.DataFrame({'id': train['id'], 'addicted_label': y, 'prediction': oof_preds})
oof_df.to_csv('C:/Users/roone.DESKTOP-QK3UG2M/Downloads/projects/ModelSwarm/competitions/s6e8/data/oof_exp007.csv', index=False)
print(f"OOF saved: oof_exp007.csv")

print(f"\nDone! OOF={oof_auc:.6f}, Champion=0.95965, Delta={oof_auc-0.95965:+.6f}")
