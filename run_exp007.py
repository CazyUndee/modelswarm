import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('competitions/s6e8/data/train.csv')
test = pd.read_csv('competitions/s6e8/data/test.csv')

print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')

# Feature engineering
def add_features(df):
    df = df.copy()
    df['social_ratio'] = df['social_media_hours'] / (df['daily_screen_time_hours'] + 0.01)
    df['gaming_ratio'] = df['gaming_hours'] / (df['daily_screen_time_hours'] + 0.01)
    df['weekend_boost'] = df['weekend_screen_time'] / (df['daily_screen_time_hours'] + 0.01)
    df['sleep_debt'] = 8 - df['sleep_hours']
    df['total_leisure'] = df['social_media_hours'] + df['gaming_hours']
    df['screen_notif'] = df['daily_screen_time_hours'] * df['notifications_per_day']
    return df

train = add_features(train)
test = add_features(test)

# Features to use
cat_features = ['gender', 'stress_level', 'academic_work_impact']
num_features = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
                'work_study_hours', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day',
                'weekend_screen_time', 'social_ratio', 'gaming_ratio', 'weekend_boost',
                'sleep_debt', 'total_leisure', 'screen_notif']

all_features = num_features + cat_features

X = train[all_features].copy()
y = train['addicted_label']
X_test = test[all_features].copy()

# Handle categoricals
for col in cat_features:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store OOF predictions
oof_lgb = np.zeros(len(train))
oof_xgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))

# Store test predictions
test_lgb = np.zeros(len(test))
test_xgb = np.zeros(len(test))
test_cat = np.zeros(len(test))

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train, categorical_feature=cat_features)
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    test_lgb += lgb_model.predict_proba(X_test)[:, 1] / 5
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        early_stopping_rounds=50, random_state=42, verbosity=0,
        enable_categorical=True
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    test_xgb += xgb_model.predict_proba(X_test)[:, 1] / 5
    
    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6,
        random_seed=42, verbose=0, cat_features=cat_features
    )
    cat_model.fit(X_train, y_train)
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    test_cat += cat_model.predict_proba(X_test)[:, 1] / 5
    
    # Fold scores
    auc_lgb = roc_auc_score(y_val, oof_lgb[val_idx])
    auc_xgb = roc_auc_score(y_val, oof_xgb[val_idx])
    auc_cat = roc_auc_score(y_val, oof_cat[val_idx])
    
    fold_scores.append({'fold': fold, 'lgb': auc_lgb, 'xgb': auc_xgb, 'cat': auc_cat})
    print(f'Fold {fold}: LGB={auc_lgb:.5f} | XGB={auc_xgb:.5f} | CAT={auc_cat:.5f}')

# Overall OOF scores
overall_lgb = roc_auc_score(y, oof_lgb)
overall_xgb = roc_auc_score(y, oof_xgb)
overall_cat = roc_auc_score(y, oof_cat)

print(f'\n=== Overall OOF Scores ===')
print(f'LightGBM: {overall_lgb:.5f}')
print(f'XGBoost:  {overall_xgb:.5f}')
print(f'CatBoost: {overall_cat:.5f}')

# Weighted ensemble (using OOF scores as weights)
total = overall_lgb + overall_xgb + overall_cat
w_lgb = overall_lgb / total
w_xgb = overall_xgb / total
w_cat = overall_cat / total

oof_ensemble = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cat * oof_cat
overall_ensemble = roc_auc_score(y, oof_ensemble)

print(f'\n=== Weighted Ensemble ===')
print(f'Weights: LGB={w_lgb:.3f}, XGB={w_xgb:.3f}, CAT={w_cat:.3f}')
print(f'Ensemble OOF: {overall_ensemble:.5f}')
print(f'Improvement over champion (0.95965): {overall_ensemble - 0.95965:.5f}')

# Test predictions
test_ensemble = w_lgb * test_lgb + w_xgb * test_xgb + w_cat * test_cat

# Save submission
submission = pd.DataFrame({'id': test['id'], 'addiction': test_ensemble})
submission.to_csv('competitions/s6e8/data/submission_ensemble.csv', index=False)
print(f'\nSaved submission with {len(submission)} rows')

# Save results
results = {
    'oof_scores': {'lgb': overall_lgb, 'xgb': overall_xgb, 'cat': overall_cat, 'ensemble': overall_ensemble},
    'weights': {'lgb': w_lgb, 'xgb': w_xgb, 'cat': w_cat},
    'fold_scores': fold_scores,
    'champion_score': 0.95965,
    'improvement': overall_ensemble - 0.95965
}
import json
with open('competitions/s6e8/data/ensemble_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved results JSON')
