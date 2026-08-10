import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

# Simple train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# LightGBM - fast
lgb_model = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.1, num_leaves=31,
    random_state=42, verbose=-1
)
lgb_model.fit(X_train, y_train, categorical_feature=cat_features)

val_pred = lgb_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f'LightGBM Val AUC: {val_auc:.5f}')

# Test prediction
test_pred = lgb_model.predict_proba(X_test)[:, 1]

# Save submission
submission = pd.DataFrame({'id': test['id'], 'addiction': test_pred})
submission.to_csv('competitions/s6e8/data/submission_lgb_only.csv', index=False)
print(f'Saved submission with {len(submission)} rows')
print(f'Champion score: 0.95965')
print(f'Current score: {val_auc:.5f}')
print(f'Difference: {val_auc - 0.95965:.5f}')
