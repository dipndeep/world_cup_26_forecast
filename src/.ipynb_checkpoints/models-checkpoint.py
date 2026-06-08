"""
Model Wrappers untuk prediksi pertandingan sepak bola.

Menyediakan interface seragam untuk berbagai model:
- Poisson Regression (prediksi skor)
- Classification models (prediksi hasil: W/D/L)
- Ensemble / Stacking
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PoissonMatchPredictor:
    """
    Prediksi pertandingan menggunakan model Poisson.
    
    Memprediksi jumlah gol masing-masing tim secara independen,
    lalu menghitung probabilitas menang/seri/kalah dari matriks skor.
    """
    
    def __init__(self, max_goals: int = 10):
        """
        Args:
            max_goals: Maksimum gol yang diperhitungkan dalam matriks skor
        """
        self.max_goals = max_goals
        self.home_model = None
        self.away_model = None
        self.feature_columns = None
    
    def fit(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series,
            feature_columns: Optional[List[str]] = None):
        """
        Fit model Poisson regression.
        
        Args:
            X: Feature matrix
            y_home: Home goals
            y_away: Away goals
            feature_columns: Kolom fitur yang digunakan
        """
        import statsmodels.api as sm
        
        self.feature_columns = feature_columns or X.columns.tolist()
        X_features = sm.add_constant(X[self.feature_columns].astype(float))
        
        # Fit Poisson GLM untuk home goals
        self.home_model = sm.GLM(y_home.astype(float), X_features, 
                                  family=sm.families.Poisson()).fit(disp=False)
        
        # Fit Poisson GLM untuk away goals
        self.away_model = sm.GLM(y_away.astype(float), X_features,
                                  family=sm.families.Poisson()).fit(disp=False)
        
        return self
    
    def predict_goals(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediksi expected goals untuk home dan away team.
        
        Returns:
            Tuple (lambda_home, lambda_away) — expected goals
        """
        import statsmodels.api as sm
        X_features = sm.add_constant(X[self.feature_columns].astype(float))
        
        lambda_home = self.home_model.predict(X_features)
        lambda_away = self.away_model.predict(X_features)
        
        return lambda_home.values, lambda_away.values
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prediksi probabilitas [P(Home Win), P(Draw), P(Away Win)].
        
        Returns:
            Array shape (n_samples, 3)
        """
        lambda_home, lambda_away = self.predict_goals(X)
        
        probas = []
        for lh, la in zip(lambda_home, lambda_away):
            p_home_win, p_draw, p_away_win = self._score_matrix_probs(lh, la)
            probas.append([p_home_win, p_draw, p_away_win])
        
        return np.array(probas)
    
    def _score_matrix_probs(self, lambda_home: float, 
                             lambda_away: float) -> Tuple[float, float, float]:
        """
        Hitung probabilitas dari matriks skor Poisson.
        """
        p_home_win = 0.0
        p_draw = 0.0
        p_away_win = 0.0
        
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                p = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                if i > j:
                    p_home_win += p
                elif i == j:
                    p_draw += p
                else:
                    p_away_win += p
        
        # Normalisasi
        total = p_home_win + p_draw + p_away_win
        return p_home_win / total, p_draw / total, p_away_win / total
    
    def predict_score(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prediksi skor paling mungkin.
        
        Returns:
            DataFrame dengan kolom: home_goals, away_goals, p_score
        """
        lambda_home, lambda_away = self.predict_goals(X)
        
        results = []
        for lh, la in zip(lambda_home, lambda_away):
            best_score = None
            best_prob = 0
            for i in range(self.max_goals + 1):
                for j in range(self.max_goals + 1):
                    p = poisson.pmf(i, lh) * poisson.pmf(j, la)
                    if p > best_prob:
                        best_prob = p
                        best_score = (i, j)
            results.append({
                'pred_home_goals': best_score[0],
                'pred_away_goals': best_score[1],
                'p_score': best_prob,
            })
        
        return pd.DataFrame(results)


class ClassificationPredictor:
    """
    Wrapper untuk model klasifikasi (Random Forest, XGBoost, Logistic Regression).
    
    Target: 0 = Away Win, 1 = Draw, 2 = Home Win
    """
    
    RESULT_MAP = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Args:
            model_type: 'random_forest', 'xgboost', 'logistic_regression'
            **kwargs: Hyperparameters untuk model
        """
        self.model_type = model_type
        self.model = self._create_model(model_type, **kwargs)
        self.feature_columns = None
    
    def _create_model(self, model_type: str, **kwargs):
        if model_type == 'random_forest':
            defaults = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_leaf': 20,
                'random_state': 42,
                'n_jobs': -1,
            }
            defaults.update(kwargs)
            return RandomForestClassifier(**defaults)
        
        elif model_type == 'xgboost':
            from xgboost import XGBClassifier
            defaults = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'mlogloss',
            }
            defaults.update(kwargs)
            return XGBClassifier(**defaults)
        
        elif model_type == 'logistic_regression':
            defaults = {
                'multi_class': 'multinomial',
                'max_iter': 1000,
                'random_state': 42,
                'C': 1.0,
            }
            defaults.update(kwargs)
            return LogisticRegression(**defaults)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def encode_result(home_goals: pd.Series, away_goals: pd.Series) -> pd.Series:
        """Encode hasil pertandingan: 0=Away Win, 1=Draw, 2=Home Win."""
        result = pd.Series(1, index=home_goals.index)  # Default: Draw
        result[home_goals > away_goals] = 2   # Home Win
        result[home_goals < away_goals] = 0   # Away Win
        return result
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            feature_columns: Optional[List[str]] = None):
        """Fit model klasifikasi."""
        self.feature_columns = feature_columns or X.columns.tolist()
        X_features = X[self.feature_columns].astype(float)
        self.model.fit(X_features, y)
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prediksi probabilitas [P(Away Win), P(Draw), P(Home Win)].
        
        Returns:
            Array shape (n_samples, 3)
        """
        X_features = X[self.feature_columns].astype(float)
        return self.model.predict_proba(X_features)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prediksi kelas (0, 1, atau 2)."""
        X_features = X[self.feature_columns].astype(float)
        return self.model.predict(X_features)
    
    def feature_importance(self) -> pd.Series:
        """Return feature importance (jika tersedia)."""
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(
                self.model.feature_importances_,
                index=self.feature_columns
            ).sort_values(ascending=False)
        return pd.Series(dtype=float)


class EnsemblePredictor:
    """
    Ensemble (weighted average) dari beberapa model.
    """
    
    def __init__(self, models: List, weights: Optional[List[float]] = None):
        """
        Args:
            models: List of fitted models (harus punya predict_proba)
            weights: Bobot untuk setiap model. Jika None, rata-rata sama.
        """
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Prediksi probabilitas ensemble (weighted average)."""
        probas = np.zeros((len(X), 3))
        for model, weight in zip(self.models, self.weights):
            probas += weight * model.predict_proba(X)
        return probas


# ============================================================
# Evaluation Utilities
# ============================================================

def evaluate_model(y_true: np.ndarray, y_proba: np.ndarray, 
                   model_name: str = 'Model') -> Dict[str, float]:
    """
    Evaluasi model dengan beberapa metrik.
    
    Args:
        y_true: Label sebenarnya (0, 1, 2)
        y_proba: Probabilitas prediksi shape (n, 3)
        model_name: Nama model
    
    Returns:
        Dict berisi log_loss, accuracy, brier_score
    """
    y_pred = y_proba.argmax(axis=1)
    
    ll = log_loss(y_true, y_proba, labels=[0, 1, 2])
    acc = accuracy_score(y_true, y_pred)
    
    # Brier score per class
    brier_scores = []
    for cls in range(3):
        y_bin = (y_true == cls).astype(int)
        bs = brier_score_loss(y_bin, y_proba[:, cls])
        brier_scores.append(bs)
    avg_brier = np.mean(brier_scores)
    
    results = {
        'model': model_name,
        'log_loss': round(ll, 4),
        'accuracy': round(acc, 4),
        'brier_score': round(avg_brier, 4),
    }
    
    return results


def time_series_cv(model_class, X: pd.DataFrame, y: pd.Series,
                   feature_columns: List[str],
                   n_splits: int = 5, **model_kwargs) -> List[Dict]:
    """
    Time-series cross-validation.
    
    Returns:
        List of evaluation dicts per fold
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = model_class(**model_kwargs)
        if hasattr(model, 'feature_columns'):
            model.fit(X_train, y_train, feature_columns=feature_columns)
        else:
            model.fit(X_train[feature_columns], y_train)
        
        y_proba = model.predict_proba(X_val)
        fold_result = evaluate_model(y_val.values, y_proba, 
                                      f'Fold {fold+1}')
        results.append(fold_result)
    
    return results
