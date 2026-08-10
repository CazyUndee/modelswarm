import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
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

# Fill NaN in categoricals with 'missing' string
for col in cat_features:
    X[col] = X[col].fillna('missing').astype('category')
    X_test[col] = X_test[col].fillna('missing').astype('category')

# 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train))
test_lgb = np.zeros(len(test))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # LightGBM with more trees
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train, categorical_feature=cat_features)
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    test_lgb += lgb_model.predict_proba(X_test)[:, 1] / 5
    
    auc = roc_auc_score(y_val, oof_lgb[val_idx])
    fold_scores.append({'fold': fold, 'auc': auc})
    print(f'Fold {fold}: {auc:.5f}')

overall = roc_auc_score(y, oof_lgb)
print(f'\n=== LightGBM 5-fold CV ===')
print(f'OOF AUC: {overall:.5f}')
print(f'Champion: 0.95965')
print(f'Diff: {overall - 0.95965:.5f}')

# Save
submission = pd.DataFrame({'id': test['id'], 'addiction': test_lgb})
submission.to_csv('competitions/s6e8/data/submission_lgb_cv5.csv', index=False)
print(f'Saved submission')

# Save results
import json
results = {
    'oof_score': overall,
    'fold_scores': fold_scores,
    'champion_score': 0.95965,
    'improvement': overall - 0.95965
}
with open('competitions/s6e8/data/lgb_cv5_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved results JSON')
