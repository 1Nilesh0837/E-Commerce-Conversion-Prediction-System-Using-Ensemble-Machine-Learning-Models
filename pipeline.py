"""
=============================================================================
User Conversion Prediction Pipeline
Full Competition Pipeline: Top-1 Strategy
=============================================================================
Phases:
  1. EDA & Data Loading
  2. Missing Value Engineering (indicators + median fill)
  3. Advanced Feature Engineering (15-20 features)
  4. Target Encoding (KFold, leak-free)
  5. Frequency Encoding
  6. Model Training: CatBoost, LightGBM, XGBoost
  7. Optuna Hyperparameter Tuning
  8. Threshold Optimization
  9. Ensemble & Final Submission
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from catboost import CatBoostClassifier  # type: ignore
from lightgbm import LGBMClassifier  # type: ignore
from xgboost import XGBClassifier  # type: ignore
from sklearn.ensemble import ExtraTreesClassifier
import optuna  # type: ignore
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# PHASE 1: LOAD DATA
# ============================================================================
print("=" * 70)
print("PHASE 1: Loading Data")
print("=" * 70)

train = pd.read_csv('train.csv')
public_test = pd.read_csv('public_test.csv')
private_test = pd.read_csv('private_test.csv')

print(f"Train: {train.shape}, Public: {public_test.shape}, Private: {private_test.shape}")
print(f"Target distribution: {train['Converted'].value_counts(normalize=True).to_dict()}")

# We can use public_test labels for a richer training set OR for validation
# Strategy: combine train + public_test for maximum training data
full_train = pd.concat([train, public_test], axis=0, ignore_index=True)
print(f"Full train (train + public): {full_train.shape}")

TARGET = 'Converted'
y_full = full_train[TARGET].values

# ============================================================================
# PHASE 2: MISSING VALUE ENGINEERING
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 2: Missing Value Engineering")
print("=" * 70)

def add_missing_indicators(df):
    """Create binary flags for missing values - missingness itself can be predictive."""
    df['Age_missing'] = df['Age'].isnull().astype(int)
    df['Income_missing'] = df['Income'].isnull().astype(int)
    df['Time_missing'] = df['Time_On_Site'].isnull().astype(int)
    # Interaction: how many features are missing for this user
    df['total_missing'] = df['Age_missing'] + df['Income_missing'] + df['Time_missing']
    return df

full_train = add_missing_indicators(full_train)
private_test = add_missing_indicators(private_test)

# Fill with median (computed on train only to avoid test leakage)
train_medians = {
    'Age': train['Age'].median(),
    'Income': train['Income'].median(),
    'Time_On_Site': train['Time_On_Site'].median()
}
print(f"Fill medians: {train_medians}")

for col, med in train_medians.items():
    full_train[col] = full_train[col].fillna(med)
    private_test[col] = private_test[col].fillna(med)

print(f"Missing after fill - Train: {full_train.isnull().sum().sum()}, Test: {private_test.isnull().sum().sum()}")

# ============================================================================
# PHASE 3: ADVANCED FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 3: Advanced Feature Engineering")
print("=" * 70)

def engineer_features(df):
    """Create 15-20 powerful engineered features."""
    
    # --- Engagement Features ---
    # 1. Product-to-page ratio: how focused is the browsing?
    df['product_page_ratio'] = df['Products_Viewed'] / (df['Pages_Viewed'] + 1)
    
    # 2. Purchase intent: products viewed * time spent
    df['intent_score'] = df['Products_Viewed'] * df['Time_On_Site']
    
    # 3. Session depth: pages * time
    df['session_depth'] = df['Pages_Viewed'] * df['Time_On_Site']
    
    # 4. Browsing intensity: pages per minute of site time
    df['browsing_intensity'] = df['Pages_Viewed'] / (df['Time_On_Site'] + 0.1)
    
    # 5. Product browsing intensity: products per minute
    df['product_intensity'] = df['Products_Viewed'] / (df['Time_On_Site'] + 0.1)
    
    # --- Financial Features ---
    # 6. Historical loyalty: previous purchases weighted by income
    df['loyalty_score'] = df['Previous_Purchases'] * df['Income']
    
    # 7. Income per purchase: spending power per transaction
    df['income_purchase_ratio'] = df['Income'] / (df['Previous_Purchases'] + 1)
    
    # 8. Affordability index: income relative to age (proxy for career stage)
    df['affordability'] = df['Income'] / (df['Age'] + 1)
    
    # --- Discount Features ---
    # 9. Discount interest: saw discount AND viewed many products
    df['discount_interest'] = df['Discount_Seen'] * df['Products_Viewed']
    
    # 10. Discount with high intent
    df['discount_intent'] = df['Discount_Seen'] * df['intent_score']
    
    # --- Interaction Features ---
    # 11. Browser-age interaction
    df['browser_age'] = df['Browser_Version'] * df['Age']
    
    # 12. Age-income interaction  
    df['age_income'] = df['Age'] * df['Income']
    
    # 13. Pages squared (capture non-linearity)
    df['pages_sq'] = df['Pages_Viewed'] ** 2
    
    # 14. Products squared
    df['products_sq'] = df['Products_Viewed'] ** 2
    
    # 15. Log transforms for skewed features
    df['log_income'] = np.log1p(df['Income'])
    df['log_time'] = np.log1p(df['Time_On_Site'])
    
    # 16. Time_On_Site has extreme outliers (max 607!) - clipped version
    df['time_clipped'] = df['Time_On_Site'].clip(upper=60)
    
    # 17. High engagement flag
    df['high_engagement'] = ((df['Pages_Viewed'] > 20) & (df['Products_Viewed'] > 15)).astype(int)
    
    # 18. Returning buyer flag
    df['returning_buyer'] = (df['Previous_Purchases'] > 0).astype(int)
    
    # 19. Pages-products difference (browsing without focus?)
    df['pages_minus_products'] = df['Pages_Viewed'] - df['Products_Viewed']
    
    # 20. Engagement score composite
    df['engagement_composite'] = (
        df['Pages_Viewed'] * 0.3 + 
        df['Products_Viewed'] * 0.4 + 
        df['Time_On_Site'] * 0.1 + 
        df['Discount_Seen'] * 0.2
    )
    
    return df

full_train = engineer_features(full_train)
private_test = engineer_features(private_test)
print(f"Features after engineering: {full_train.shape[1]}")

# ============================================================================
# PHASE 4: TARGET ENCODING (KFold, leak-free)
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 4: Target Encoding (KFold)")
print("=" * 70)

target_encode_cols = ['Device_Type', 'Traffic_Source', 'Campaign_Code', 'Browser_Version', 'City_Tier']

def kfold_target_encode(train_df, test_df, cols, target, n_splits=5, seed=42):
    """KFold target encoding to avoid leakage."""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for col in cols:
        enc_col = f'{col}_te'
        train_df[enc_col] = 0.0
        
        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(train_df, train_df[target])):
            # Compute means on training fold only
            means = train_df.iloc[tr_idx].groupby(col)[target].mean()
            # Apply to validation fold
            train_df.loc[train_df.index[val_idx], enc_col] = train_df.iloc[val_idx][col].map(means)
        
        # Fill any NaN with global mean
        global_mean = train_df[target].mean()
        train_df[enc_col] = train_df[enc_col].fillna(global_mean)
        
        # For test: use full train means
        full_means = train_df.groupby(col)[target].mean()
        test_df[enc_col] = test_df[col].map(full_means).fillna(global_mean)
        
        print(f"  {col} -> {enc_col} (range: {train_df[enc_col].min():.3f} - {train_df[enc_col].max():.3f})")
    
    return train_df, test_df

full_train, private_test = kfold_target_encode(full_train, private_test, target_encode_cols, TARGET)

# ============================================================================
# PHASE 5: FREQUENCY ENCODING
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 5: Frequency Encoding")
print("=" * 70)

freq_encode_cols = ['Campaign_Code', 'Browser_Version']

for col in freq_encode_cols:
    freq = full_train[col].value_counts()
    enc_col = f'{col}_freq'
    full_train[enc_col] = full_train[col].map(freq)
    private_test[enc_col] = private_test[col].map(freq).fillna(0)
    print(f"  {col} -> {enc_col} (max freq: {full_train[enc_col].max()})")

# ============================================================================
# PREPARE FINAL FEATURE MATRIX
# ============================================================================
print("\n" + "=" * 70)
print("Preparing Final Feature Matrix")
print("=" * 70)

# Drop ID, target, and raw categoricals
drop_cols = ['User_ID', TARGET, 'Device_Type', 'Traffic_Source']
feature_cols = [c for c in full_train.columns if c not in drop_cols]

X_full = full_train[feature_cols].values
y_full = full_train[TARGET].values
X_private = private_test[[c for c in feature_cols if c in private_test.columns]].values

print(f"Feature matrix: {X_full.shape}")
print(f"Private test: {X_private.shape}")
print(f"Feature names: {feature_cols}")

# ============================================================================
# PHASE 6: CROSS VALIDATION WITH MULTIPLE MODELS
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 6: 5-Fold Stratified Cross Validation")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model_fn, X, y, model_name, skf):
    """Evaluate a model with stratified k-fold CV and return OOF predictions."""
    oof_preds = np.zeros(len(y))
    fold_scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        model = model_fn()
        
        if model_name == 'CatBoost':
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        elif model_name == 'LightGBM':
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        elif model_name == 'XGBoost':
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        else:
            model.fit(X_tr, y_tr)
        
        if hasattr(model, 'predict_proba'):
            val_proba = model.predict_proba(X_val)[:, 1]
        else:
            val_proba = model.predict(X_val)
        
        oof_preds[val_idx] = val_proba
        
        # Use default 0.5 threshold for initial eval
        val_pred = (val_proba >= 0.5).astype(int)
        score = f1_score(y_val, val_pred)
        fold_scores.append(score)
        print(f"  Fold {fold+1}: F1 = {score:.5f}")
    
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    print(f"  >> {model_name} CV F1: {mean_f1:.5f} (+/- {std_f1:.5f})")
    return oof_preds, mean_f1

# ----- CatBoost -----
print("\n--- CatBoost ---")
def catboost_fn():
    return CatBoostClassifier(
        iterations=1500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        bagging_temperature=0.8,
        random_strength=1.0,
        loss_function='Logloss',
        eval_metric='F1',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100
    )

cat_oof, cat_cv = evaluate_model(catboost_fn, X_full, y_full, 'CatBoost', skf)

# ----- LightGBM -----
print("\n--- LightGBM ---")
def lgbm_fn():
    return LGBMClassifier(
        n_estimators=1500,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=40,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        objective='binary',
        metric='binary_logloss',
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )

lgb_oof, lgb_cv = evaluate_model(lgbm_fn, X_full, y_full, 'LightGBM', skf)

# ----- XGBoost -----
print("\n--- XGBoost ---")
def xgb_fn():
    return XGBClassifier(
        n_estimators=1500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
        early_stopping_rounds=100,
        n_jobs=-1,
    )

xgb_oof, xgb_cv = evaluate_model(xgb_fn, X_full, y_full, 'XGBoost', skf)

# ----- ExtraTrees -----
print("\n--- ExtraTrees ---")
def et_fn():
    return ExtraTreesClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

et_oof, et_cv = evaluate_model(et_fn, X_full, y_full, 'ExtraTrees', skf)

# ============================================================================
# PHASE 7: OPTUNA HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 7: Optuna Hyperparameter Tuning")
print("=" * 70)

def optuna_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 2500),
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 2.0),
        'random_strength': trial.suggest_float('random_strength', 0.1, 3.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'loss_function': 'Logloss',
        'eval_metric': 'F1',
        'random_seed': 42,
        'verbose': 0,
        'early_stopping_rounds': 100,
    }
    
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        model = CatBoostClassifier(**params)
        model.fit(X_full[tr_idx], y_full[tr_idx], 
                  eval_set=(X_full[val_idx], y_full[val_idx]), verbose=0)
        val_proba = model.predict_proba(X_full[val_idx])[:, 1]
        # Optimize threshold too during tuning
        best_f1 = 0
        for t in np.arange(0.30, 0.60, 0.01):
            f1 = f1_score(y_full[val_idx], (val_proba >= t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
        scores.append(best_f1)
    return np.mean(scores)

print("Tuning CatBoost (20 trials)...")
study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(optuna_catboost, n_trials=20, show_progress_bar=False)
print(f"Best CatBoost CV F1: {study_cat.best_value:.5f}")
print(f"Best params: {study_cat.best_params}")

def optuna_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2500),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 80),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'objective': 'binary',
        'metric': 'binary_logloss',
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        model = LGBMClassifier(**params)
        model.fit(X_full[tr_idx], y_full[tr_idx], 
                  eval_set=[(X_full[val_idx], y_full[val_idx])])
        val_proba = model.predict_proba(X_full[val_idx])[:, 1]
        best_f1 = 0
        for t in np.arange(0.30, 0.60, 0.01):
            f1 = f1_score(y_full[val_idx], (val_proba >= t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
        scores.append(best_f1)
    return np.mean(scores)

print("\nTuning LightGBM (20 trials)...")
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(optuna_lgbm, n_trials=20, show_progress_bar=False)
print(f"Best LightGBM CV F1: {study_lgb.best_value:.5f}")
print(f"Best params: {study_lgb.best_params}")

def optuna_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2500),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 2.0),
        'eval_metric': 'logloss',
        'random_state': 42,
        'verbosity': 0,
        'early_stopping_rounds': 100,
        'n_jobs': -1,
    }
    
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        model = XGBClassifier(**params)
        model.fit(X_full[tr_idx], y_full[tr_idx],
                  eval_set=[(X_full[val_idx], y_full[val_idx])], verbose=0)
        val_proba = model.predict_proba(X_full[val_idx])[:, 1]
        best_f1 = 0
        for t in np.arange(0.30, 0.60, 0.01):
            f1 = f1_score(y_full[val_idx], (val_proba >= t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
        scores.append(best_f1)
    return np.mean(scores)

print("\nTuning XGBoost (20 trials)...")
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(optuna_xgb, n_trials=20, show_progress_bar=False)
print(f"Best XGBoost CV F1: {study_xgb.best_value:.5f}")
print(f"Best params: {study_xgb.best_params}")

# ============================================================================
# PHASE 8 & 9: RETRAIN WITH BEST PARAMS, ENSEMBLE, THRESHOLD OPTIMIZE
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 8-9: Final Ensemble with Tuned Models + Threshold Optimization")
print("=" * 70)

# Retrain tuned models with OOF predictions for threshold search
def retrain_tuned(study, model_class, model_name, X, y, X_test, skf, extra_params=None):
    """Retrain tuned model, collect OOF preds, and predict on test."""
    best_params = study.best_params.copy()
    if extra_params:
        best_params.update(extra_params)
    
    oof = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        model = model_class(**best_params)
        
        if model_name == 'CatBoost':
            model.fit(X[tr_idx], y[tr_idx], eval_set=(X[val_idx], y[val_idx]), verbose=0)
        elif model_name == 'LightGBM':
            model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])])
        elif model_name == 'XGBoost':
            model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=0)
        
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
        
        f1 = f1_score(y[val_idx], (oof[val_idx] >= 0.5).astype(int))
        print(f"  {model_name} Fold {fold+1}: F1 = {f1:.5f}")
    
    return oof, test_preds

# CatBoost tuned
print("\n--- Tuned CatBoost ---")
cat_extra = {'loss_function': 'Logloss', 'eval_metric': 'F1', 'random_seed': 42, 'verbose': 0, 'early_stopping_rounds': 100}
cat_oof_tuned, cat_test = retrain_tuned(study_cat, CatBoostClassifier, 'CatBoost', X_full, y_full, X_private, skf, cat_extra)

# LightGBM tuned
print("\n--- Tuned LightGBM ---")
lgb_extra = {'objective': 'binary', 'metric': 'binary_logloss', 'random_state': 42, 'verbose': -1, 'n_jobs': -1}
lgb_oof_tuned, lgb_test = retrain_tuned(study_lgb, LGBMClassifier, 'LightGBM', X_full, y_full, X_private, skf, lgb_extra)

# XGBoost tuned
print("\n--- Tuned XGBoost ---")
xgb_extra = {'eval_metric': 'logloss', 'random_state': 42, 'verbosity': 0, 'early_stopping_rounds': 100, 'n_jobs': -1}
xgb_oof_tuned, xgb_test = retrain_tuned(study_xgb, XGBClassifier, 'XGBoost', X_full, y_full, X_private, skf, xgb_extra)

# ============================================================================
# ENSEMBLE + WEIGHT OPTIMIZATION
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 10: Ensemble Weight & Threshold Optimization")
print("=" * 70)

# Search best ensemble weights
best_ensemble_f1 = 0
best_weights = (0.4, 0.3, 0.3)
best_threshold = 0.5

for w1 in np.arange(0.2, 0.7, 0.05):
    for w2 in np.arange(0.1, 0.6, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05:
            continue
        ensemble_oof = w1 * cat_oof_tuned + w2 * lgb_oof_tuned + w3 * xgb_oof_tuned
        
        for t in np.arange(0.25, 0.65, 0.005):
            preds = (ensemble_oof >= t).astype(int)
            f1 = f1_score(y_full, preds)
            if f1 > best_ensemble_f1:
                best_ensemble_f1 = f1
                best_weights = (round(w1, 2), round(w2, 2), round(w3, 2))
                best_threshold = round(t, 3)

print(f"Best Ensemble Weights: CatBoost={best_weights[0]}, LightGBM={best_weights[1]}, XGBoost={best_weights[2]}")
print(f"Best Threshold: {best_threshold}")
print(f"Best Ensemble CV F1: {best_ensemble_f1:.5f}")

# Also check individual model thresholds
for name, oof in [('CatBoost', cat_oof_tuned), ('LightGBM', lgb_oof_tuned), ('XGBoost', xgb_oof_tuned)]:
    best_f1_single = 0
    best_t_single = 0.5
    for t in np.arange(0.25, 0.65, 0.005):
        f1 = f1_score(y_full, (oof >= t).astype(int))
        if f1 > best_f1_single:
            best_f1_single = f1
            best_t_single = t
    print(f"  {name} alone: Best F1={best_f1_single:.5f} at threshold={best_t_single:.3f}")

# ============================================================================
# FINAL PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("Generating Final Submission")
print("=" * 70)

# Ensemble test predictions
final_proba = (
    best_weights[0] * cat_test + 
    best_weights[1] * lgb_test + 
    best_weights[2] * xgb_test
)

final_preds = (final_proba >= best_threshold).astype(int)

print(f"Prediction distribution: {pd.Series(final_preds).value_counts().to_dict()}")
print(f"Positive rate: {final_preds.mean():.4f} (train was {y_full.mean():.4f})")

# Create submission
submission = pd.DataFrame({
    'User_ID': private_test['User_ID'],
    'Converted': final_preds
})

submission.to_csv('submission.csv', index=False)
print(f"\nsubmission.csv saved! Shape: {submission.shape}")
print(submission.head(10))

# Also save individual model submissions for backup
for name, test_proba in [('catboost', cat_test), ('lightgbm', lgb_test), ('xgboost', xgb_test)]:
    # Use optimized threshold for each
    best_t = 0.5
    best_f1_b = 0
    oof_map = {'catboost': cat_oof_tuned, 'lightgbm': lgb_oof_tuned, 'xgboost': xgb_oof_tuned}
    for t in np.arange(0.25, 0.65, 0.005):
        f1 = f1_score(y_full, (oof_map[name] >= t).astype(int))
        if f1 > best_f1_b:
            best_f1_b = f1
            best_t = t
    
    sub = pd.DataFrame({
        'User_ID': private_test['User_ID'],
        'Converted': (test_proba >= best_t).astype(int)
    })
    sub.to_csv(f'submission_{name}.csv', index=False)
    print(f"  submission_{name}.csv saved (threshold={best_t:.3f})")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE!")
print("=" * 70)
print(f"""
Summary:
  - Features engineered: {len(feature_cols)}
  - Models: CatBoost + LightGBM + XGBoost (all Optuna-tuned)
  - Ensemble weights: {best_weights}
  - Optimal threshold: {best_threshold}
  - CV F1 Score: {best_ensemble_f1:.5f}
  
Files created:
  - submission.csv (ensemble - PRIMARY)
  - submission_catboost.csv
  - submission_lightgbm.csv  
  - submission_xgboost.csv
""")
