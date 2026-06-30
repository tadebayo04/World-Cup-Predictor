import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from features import FEATURE_COLS

RANDOM_STATE = 42
TEST_CUTOFF = '2022-01-01'

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
_PROCESSED = os.path.join(_DATA_DIR, 'processed')
_MODELS_DIR = os.path.join(_DATA_DIR, 'models')


def load_model_ready():
    """Load model_ready.csv and future_fixtures_features.csv."""
    train = pd.read_csv(os.path.join(_PROCESSED, 'model_ready.csv'))
    future = pd.read_csv(os.path.join(_PROCESSED, 'future_fixtures_features.csv'))
    future_meta = pd.read_csv(os.path.join(_PROCESSED, 'future_fixtures.csv'))
    return train, future, future_meta


def time_based_split(df, test_cutoff=TEST_CUTOFF):
    """Split on date rather than random to prevent temporal leakage.

    Keeps 2022 Qatar World Cup and 2022-2026 qualifiers in the test set,
    giving a realistic evaluation of World Cup prediction performance.
    """
    dates = pd.to_datetime(df['date'], errors='coerce')
    mask = dates >= test_cutoff

    X = df[FEATURE_COLS]
    y = df['outcome']

    X_train = X[~mask].reset_index(drop=True)
    X_test  = X[mask].reset_index(drop=True)
    y_train = y[~mask].reset_index(drop=True)
    y_test  = y[mask].reset_index(drop=True)

    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"Test class distribution:\n{y_test.value_counts()}")
    return X_train, X_test, y_train, y_test


def preprocess_features(X_train, X_test, X_future=None):
    """Return scaled versions (for LR) and unscaled versions (for RF).

    Squad value columns are log-transformed before scaling to handle
    extreme right skew (max ~3.4B EUR vs median ~1.5M EUR).
    The scaler is fit on training data only and applied to test/future.
    """
    log_cols = ['home_squad_value_log', 'away_squad_value_log']

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLS)

    result = [X_train_scaled, X_test_scaled, X_train, X_test, scaler]

    if X_future is not None:
        X_future_scaled = pd.DataFrame(scaler.transform(X_future[FEATURE_COLS]), columns=FEATURE_COLS)
        result.append(X_future_scaled)
        result.append(X_future[FEATURE_COLS])

    return result


def train_random_forest(X_train, y_train):
    """Train a RandomForestClassifier with GridSearchCV over key hyperparameters.

    Uses TimeSeriesSplit to avoid future leakage during cross-validation.
    class_weight='balanced' corrects for the Home Win majority class.
    """
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 5],
    }
    base = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    cv = TimeSeriesSplit(n_splits=5)
    search = GridSearchCV(base, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    print(f"RF best params: {search.best_params_}  |  CV f1_macro: {search.best_score_:.4f}")
    return search.best_estimator_


def train_logistic_regression(X_train_scaled, y_train):
    """Train a multinomial LogisticRegression with GridSearchCV over C.

    Uses TimeSeriesSplit and balanced class weights to handle class imbalance.
    """
    param_grid = {'C': [0.01, 0.1, 1.0, 10.0]}
    base = LogisticRegression(
        solver='lbfgs', class_weight='balanced',
        max_iter=1000, random_state=RANDOM_STATE,
    )
    cv = TimeSeriesSplit(n_splits=5)
    search = GridSearchCV(base, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1)
    search.fit(X_train_scaled, y_train)
    print(f"LR best C: {search.best_params_}  |  CV f1_macro: {search.best_score_:.4f}")
    return search.best_estimator_


def evaluate_model(model, X_test, y_test, model_name):
    """Print and return evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_per_class = f1_score(y_test, y_pred, average=None, labels=['Home Win', 'Away Win', 'Draw'])
    cm = confusion_matrix(y_test, y_pred, labels=['Home Win', 'Away Win', 'Draw'])

    print(f"\n{'='*50}")
    print(f"{model_name} — Test Results")
    print(f"{'='*50}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 macro:  {f1_macro:.4f}")
    print(f"F1 per class: Home Win={f1_per_class[0]:.3f}  Away Win={f1_per_class[1]:.3f}  Draw={f1_per_class[2]:.3f}")
    print(classification_report(y_test, y_pred, labels=['Home Win', 'Away Win', 'Draw']))

    return {
        'model_name': model_name,
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_home_win': f1_per_class[0],
        'f1_away_win': f1_per_class[1],
        'f1_draw': f1_per_class[2],
        'confusion_matrix': cm,
    }


def save_model(model, name):
    os.makedirs(_MODELS_DIR, exist_ok=True)
    path = os.path.join(_MODELS_DIR, f'{name}.pkl')
    joblib.dump(model, path)
    print(f"Saved model → {path}")


def predict_fixtures(rf, lr, X_future_rf, X_future_lr, future_meta,
                     output_filename='predictions.csv'):
    """Generate predictions for upcoming fixtures from both models.

    Returns a DataFrame with match metadata plus per-model predicted class
    and win/draw probabilities.
    """
    classes = rf.classes_  # e.g. ['Away Win', 'Draw', 'Home Win']
    hw_idx = list(classes).index('Home Win')
    aw_idx = list(classes).index('Away Win')
    dr_idx = list(classes).index('Draw')

    rf_proba = rf.predict_proba(X_future_rf)
    lr_proba = lr.predict_proba(X_future_lr)

    results = future_meta[['date', 'home_team', 'away_team']].copy()
    results['rf_prediction'] = rf.predict(X_future_rf)
    results['rf_home_win_prob'] = rf_proba[:, hw_idx]
    results['rf_away_win_prob'] = rf_proba[:, aw_idx]
    results['rf_draw_prob'] = rf_proba[:, dr_idx]

    results['lr_prediction'] = lr.predict(X_future_lr)
    results['lr_home_win_prob'] = lr_proba[:, hw_idx]
    results['lr_away_win_prob'] = lr_proba[:, aw_idx]
    results['lr_draw_prob'] = lr_proba[:, dr_idx]

    out_path = os.path.join(_PROCESSED, output_filename)
    results.to_csv(out_path, index=False)
    print(f"Predictions saved → {out_path}  ({len(results)} fixtures)")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train models and predict fixtures')
    parser.add_argument('--model-suffix', default='',
                        help='Suffix appended to saved model filenames (e.g. "_v2")')
    parser.add_argument('--output', default='predictions.csv',
                        help='Output filename for fixture predictions (default: predictions.csv)')
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    train_df, future_df, future_meta = load_model_ready()

    # ── Split ──────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = time_based_split(train_df)
    X_future = future_df[FEATURE_COLS]

    # ── Preprocess (scaled for LR, unscaled for RF) ────────────────────────────
    X_train_lr, X_test_lr, X_train_rf, X_test_rf, scaler, X_future_lr, X_future_rf = (
        preprocess_features(X_train, X_test, X_future)
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    print("\nTraining Random Forest...")
    rf = train_random_forest(X_train_rf, y_train)

    print("\nTraining Logistic Regression...")
    lr = train_logistic_regression(X_train_lr, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    rf_metrics = evaluate_model(rf, X_test_rf, y_test, 'Random Forest')
    lr_metrics = evaluate_model(lr, X_test_lr, y_test, 'Logistic Regression')

    # ── Save models ────────────────────────────────────────────────────────────
    suffix = args.model_suffix
    save_model(rf,     f'random_forest{suffix}')
    save_model(lr,     f'logistic_regression{suffix}')
    save_model(scaler, f'scaler{suffix}')

    # ── Predict fixtures ───────────────────────────────────────────────────────
    predictions = predict_fixtures(rf, lr, X_future_rf, X_future_lr, future_meta,
                                   output_filename=args.output)
    print("\nSample predictions:")
    print(predictions[['home_team', 'away_team', 'rf_prediction', 'lr_prediction']].head(10).to_string(index=False))
