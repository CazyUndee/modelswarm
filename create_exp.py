from modelswarm.client import Client

c = Client()

# Create cross-model ensemble experiment
exp = c.create_experiment(
    competition_id='playground-series-s6e8',
    phase=4,
    hypothesis='Cross-model ensemble: LightGBM + XGBoost + CatBoost with tuned hyperparameters on champion feature set (median imputation + missing indicators + engineered ratios) will outperform single-model LightGBM baseline through model diversity',
    model='ensemble',
    configuration='{"n_folds": 5, "models": ["lightgbm", "xgboost", "catboost"], "blend_method": "weighted_average", "lightgbm_params": {"n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 31}, "xgboost_params": {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "early_stopping": 50}, "catboost_params": {"iterations": 1000, "learning_rate": 0.05, "depth": 6}}',
    features='["age","daily_screen_time_hours","social_media_hours","gaming_hours","work_study_hours","sleep_hours","notifications_per_day","app_opens_per_day","weekend_screen_time","gender","stress_level","academic_work_impact","social_ratio","gaming_ratio","weekend_boost","sleep_debt","total_leisure"]',
    dataset='cazyundee/PlaygroundS6E8',
    validation_protocol='5-fold Stratified CV'
)

print('Created experiment:')
print(exp)
