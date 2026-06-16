"""
Win Propensity Model for RTB (Option A: Separate Model)

Estimates P(Win|X, bid) using:
- LightGBM classifier with cross-fitting
- Isotonic calibration for well-calibrated probabilities
- Wilson confidence intervals for uncertainty quantification

This approach treats win propensity estimation as a separate task,
allowing for specialized model tuning and feature engineering.
"""

from pathlib import Path
from typing import NamedTuple, Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    LOGISTIC_AVAILABLE = True
except ImportError:
    LOGISTIC_AVAILABLE = False


class WinPropensityConfig(NamedTuple):
    """Win propensity model configuration."""
    model_type: str = "lgb"  # 'lgb' or 'logistic'
    n_folds: int = 5
    clip_range: Tuple[float, float] = (0.01, 0.99)
    calibrate: bool = True
    random_seed: int = 42
    parallel_folds: int = 1  # fold 병렬화: 1=순차, -1=전체 병렬, N=N개 병렬
    # LightGBM parameters
    lgb_params: Dict[str, Any] = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_samples": 20,
        "verbose": -1,
        "n_jobs": -1,
    }
    # Logistic regression parameters
    logistic_params: Dict[str, Any] = {
        "max_iter": 1000,
        "solver": "lbfgs",
        "C": 1.0,
    }


class WinPropensityResult(NamedTuple):
    """Result of win propensity estimation."""
    propensity: np.ndarray  # P(Win|X, bid)
    propensity_clipped: np.ndarray  # Clipped to [eps, 1-eps]
    weights: np.ndarray  # IPW weights: win / P(Win)
    auc: float
    brier_score: float
    calibration_error: float


def _fit_single_fold(
    X: np.ndarray,
    win: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: WinPropensityConfig,
) -> Tuple[Any, Optional[IsotonicRegression], np.ndarray, np.ndarray]:
    """Fit a single cross-fitting fold (picklable for joblib).

    Returns:
        Tuple of (model, calibrator, val_idx, prob_val)
    """
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = win[train_idx], win[val_idx]

    if config.model_type == "lgb" and LGB_AVAILABLE:
        model = lgb.LGBMClassifier(**config.lgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )
    else:
        model = LogisticRegression(**config.logistic_params)
        model.fit(X_train, y_train)

    prob_val = model.predict_proba(X_val)[:, 1]

    calibrator = None
    if config.calibrate:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(prob_val, y_val)
        prob_val = calibrator.transform(prob_val)

    return model, calibrator, val_idx, prob_val


class WinPropensityModel:
    """Win Propensity Model with Cross-Fitting and Calibration.

    Uses K-fold cross-fitting to avoid overfitting on propensity scores:
    1. Split data into K folds
    2. For each fold, train on K-1 folds and predict on held-out fold
    3. Combine predictions for full dataset
    4. Optionally apply isotonic calibration

    This ensures that propensity scores are not overfit to training data,
    which is crucial for unbiased IPW estimation.
    """
    
    def __init__(self, config: Optional[WinPropensityConfig] = None, feature_names: Optional[List[str]] = None):
        """
        Args:
            config: Model configuration (uses defaults if None)
            feature_names: Optional list of feature names
        """
        self.config = config or WinPropensityConfig()
        self.models: List[Any] = []
        self.calibrators: List[Optional[IsotonicRegression]] = []
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = feature_names
        
    def fit(
        self,
        X: np.ndarray,
        win: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> 'WinPropensityModel':
        """Fit win propensity model using K-fold cross-fitting.
        
        Args:
            X: Feature matrix [n_samples, n_features]
            win: Win indicator [n_samples,]
            feature_names: Optional feature names for interpretability
        
        Returns:
            Self
        """
        self.feature_names = feature_names
        self.models = []
        self.calibrators = []
        
        # Cross-fitting
        kfold = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_seed,
        )
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, win)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = win[train_idx], win[val_idx]
            
            # Train model
            if self.config.model_type == "lgb" and LGB_AVAILABLE:
                model = lgb.LGBMClassifier(**self.config.lgb_params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
                )
            elif self.config.model_type == "logistic" or not LGB_AVAILABLE:
                model = LogisticRegression(**self.config.logistic_params)
                model.fit(X_train, y_train)
            else:
                raise ValueError(f"Unknown model_type: {self.config.model_type}")
            
            self.models.append(model)
            
            # Optional isotonic calibration
            if self.config.calibrate:
                probs_val = model.predict_proba(X_val)[:, 1]
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(probs_val, y_val)
                self.calibrators.append(calibrator)
            else:
                self.calibrators.append(None)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict win propensity using ensemble of K models.
        
        Args:
            X: Feature matrix [n_samples, n_features]
        
        Returns:
            Win propensities [n_samples,]
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Average predictions across all K models
        probs = np.zeros(X.shape[0])
        
        for model, calibrator in zip(self.models, self.calibrators):
            prob = model.predict_proba(X)[:, 1]
            
            if calibrator is not None:
                prob = calibrator.transform(prob)
            
            probs += prob
        
        probs /= len(self.models)
        return probs
    
    def predict_proba_crossfit(
        self,
        X: np.ndarray,
        win: np.ndarray,
    ) -> np.ndarray:
        """Predict win propensity with cross-fitting (training + prediction).

        This method fits and predicts simultaneously using cross-fitting,
        ensuring that each sample's propensity is predicted by a model
        that was not trained on that sample.

        Supports parallel fold execution via config.parallel_folds:
        - 1: sequential (default)
        - -1: use all CPU cores
        - N: use N parallel workers

        Args:
            X: Feature matrix [n_samples, n_features]
            win: Win indicator [n_samples,]

        Returns:
            Win propensities [n_samples,]
        """
        kfold = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_seed,
        )

        folds = list(kfold.split(X, win))
        n_parallel = self.config.parallel_folds

        if n_parallel == 1:
            # Sequential (original behavior)
            results = [
                _fit_single_fold(X, win, train_idx, val_idx, self.config)
                for train_idx, val_idx in folds
            ]
        else:
            from joblib import Parallel, delayed
            import os

            # Adjust per-model n_jobs to avoid CPU oversubscription
            effective_workers = n_parallel if n_parallel > 0 else os.cpu_count()
            effective_workers = min(len(folds), effective_workers)
            per_model_jobs = max(1, os.cpu_count() // effective_workers)

            # Override lgb n_jobs for parallel fold execution
            config_adj = self.config
            if self.config.model_type == "lgb":
                adj_params = dict(self.config.lgb_params)
                adj_params["n_jobs"] = per_model_jobs
                config_adj = self.config._replace(lgb_params=adj_params)

            # threading: LightGBM releases GIL, avoids huge data serialization
            results = Parallel(n_jobs=n_parallel, backend="threading", verbose=0)(
                delayed(_fit_single_fold)(X, win, train_idx, val_idx, config_adj)
                for train_idx, val_idx in folds
            )

        # Assemble results
        probs = np.zeros(X.shape[0])
        self.models = []
        self.calibrators = []

        for model, calibrator, val_idx, prob_val in results:
            self.models.append(model)
            self.calibrators.append(calibrator)
            probs[val_idx] = prob_val

        self.is_fitted = True
        return probs
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from fitted models.
        
        Returns:
            DataFrame with feature importance (None if not applicable)
        """
        if not self.is_fitted or self.config.model_type != "lgb":
            return None
        
        # Average importance across folds
        importance = np.zeros(self.models[0].n_features_)
        
        for model in self.models:
            importance += model.feature_importances_
        
        importance /= len(self.models)
        
        feature_names = self.feature_names or [f"f_{i}" for i in range(len(importance))]
        
        return pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)


def fit_win_propensity(
    X: np.ndarray,
    win: np.ndarray,
    config: Optional[WinPropensityConfig] = None,
    feature_names: Optional[List[str]] = None,
) -> Tuple[WinPropensityModel, WinPropensityResult]:
    """Fit win propensity model and return results.
    
    Convenience function that fits the model and computes all metrics.
    
    Args:
        X: Feature matrix
        win: Win indicator
        config: Model configuration
        feature_names: Optional feature names
    
    Returns:
        Tuple of (fitted model, results)
    """
    config = config or WinPropensityConfig()
    model = WinPropensityModel(config, feature_names=feature_names)

    # Cross-fit prediction
    propensity = model.predict_proba_crossfit(X, win)
    
    # Clip propensity
    propensity_clipped = np.clip(propensity, config.clip_range[0], config.clip_range[1])
    
    # Compute weights
    weights = win.astype(float) / propensity_clipped
    
    # Metrics
    auc = roc_auc_score(win, propensity)
    brier = brier_score_loss(win, propensity)
    
    # Calibration error (ECE)
    prob_true, prob_pred = calibration_curve(win, propensity, n_bins=10)
    calibration_error = np.mean(np.abs(prob_true - prob_pred))
    
    result = WinPropensityResult(
        propensity=propensity,
        propensity_clipped=propensity_clipped,
        weights=weights,
        auc=auc,
        brier_score=brier,
        calibration_error=calibration_error,
    )
    
    return model, result


def fit_win_propensity_simple(
    X: np.ndarray,
    win: np.ndarray,
    config: Optional[WinPropensityConfig] = None,
    feature_names: Optional[List[str]] = None,
) -> Tuple[WinPropensityModel, WinPropensityResult]:
    """Fit win propensity model without cross-fitting (single model).

    Trains a single model on the full dataset and predicts on the same data.
    Faster than cross-fitting but does not provide out-of-sample guarantees.
    Suitable for diagnostic notebooks where speed matters more than strict
    unbiasedness of propensity scores.

    Args:
        X: Feature matrix
        win: Win indicator
        config: Model configuration
        feature_names: Optional feature names

    Returns:
        Tuple of (fitted model, results)
    """
    config = config or WinPropensityConfig()
    model = WinPropensityModel(config, feature_names=feature_names)

    # Train single model with 80/20 split for early stopping
    n = len(win)
    indices = np.arange(n)
    rng = np.random.RandomState(config.random_seed)
    rng.shuffle(indices)
    split = int(n * 0.8)
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = win[train_idx], win[val_idx]

    if config.model_type == "lgb" and LGB_AVAILABLE:
        single_model = lgb.LGBMClassifier(**config.lgb_params)
        single_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )
    elif config.model_type == "logistic" or not LGB_AVAILABLE:
        single_model = LogisticRegression(**config.logistic_params)
        single_model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    # Optional isotonic calibration on validation set
    calibrator = None
    if config.calibrate:
        probs_val = single_model.predict_proba(X_val)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(probs_val, y_val)

    model.models = [single_model]
    model.calibrators = [calibrator]
    model.is_fitted = True

    # Predict on full data (in-sample)
    propensity = model.predict_proba(X)

    # Clip propensity
    propensity_clipped = np.clip(propensity, config.clip_range[0], config.clip_range[1])

    # Compute weights
    weights = win.astype(float) / propensity_clipped

    # Metrics
    auc = roc_auc_score(win, propensity)
    brier = brier_score_loss(win, propensity)

    # Calibration error (ECE)
    prob_true, prob_pred = calibration_curve(win, propensity, n_bins=10)
    calibration_error = np.mean(np.abs(prob_true - prob_pred))

    result = WinPropensityResult(
        propensity=propensity,
        propensity_clipped=propensity_clipped,
        weights=weights,
        auc=auc,
        brier_score=brier,
        calibration_error=calibration_error,
    )

    return model, result


def compute_win_weights(
    propensity: np.ndarray,
    win: np.ndarray,
    clip_range: Tuple[float, float] = (0.01, 0.99),
    normalize: bool = True,
) -> np.ndarray:
    """Compute IPW weights from win propensity.
    
    Args:
        propensity: Win propensity P(Win|X, bid)
        win: Win indicator
        clip_range: Propensity clipping range
        normalize: Whether to use self-normalized weights
    
    Returns:
        IPW weights
    """
    propensity_clipped = np.clip(propensity, clip_range[0], clip_range[1])
    weights = win.astype(float) / propensity_clipped
    
    if normalize:
        weight_sum = np.maximum(weights.sum(), 1e-8)
        weights = weights / weight_sum * len(weights)
    
    return weights


class WinPropensityLoadedResult(NamedTuple):
    """Result of loading pre-trained win propensity models from CLI."""
    result_win: WinPropensityResult
    importance_win: pd.DataFrame
    propensity_lr: Optional[np.ndarray]
    auc_lr: Optional[float]


class ClickPropensityLoadedResult(NamedTuple):
    """Result of loading pre-trained click propensity models from CLI."""
    result_click: WinPropensityResult
    importance_click: pd.DataFrame
    propensity_lr: Optional[np.ndarray]
    auc_lr: Optional[float]


def load_win_propensity_models(
    model_dir: Path,
    df: pd.DataFrame,
    clip_range: Tuple[float, float] = (0.01, 0.99),
) -> WinPropensityLoadedResult:
    """Load pre-trained Win PS models from CLI and predict on df.

    Loads LGB Booster from ``lgb_win.txt`` and optionally LR from
    ``lr_win.joblib``.  Returns propensity scores, metrics, and
    feature importance — matching the interface previously produced
    by in-notebook training.

    Args:
        model_dir: Directory containing trained model files.
        df: DataFrame with feature columns and 'win' label.
        clip_range: Propensity clipping range for IPW weights.

    Returns:
        WinPropensityLoadedResult with LGB (and optionally LR) outputs.
    """
    model_dir = Path(model_dir)

    # --- LGB Booster ---
    lgb_path = model_dir / "lgb_win.txt"
    if not lgb_path.exists():
        raise FileNotFoundError(
            f"LGB win model not found: {lgb_path}. "
            "Run: python scripts/train.py baseline --task win"
        )

    if not LGB_AVAILABLE:
        raise ImportError("lightgbm is required to load the LGB model")

    booster = lgb.Booster(model_file=str(lgb_path))
    feature_names = booster.feature_name()

    # Predict
    X_lgb = df[feature_names]
    propensity_win = booster.predict(X_lgb)

    # Clip + weights
    win_arr = df["win"].values
    propensity_clipped = np.clip(propensity_win, clip_range[0], clip_range[1])
    weights = win_arr.astype(float) / propensity_clipped

    # Metrics
    auc_lgb = roc_auc_score(win_arr, propensity_win)
    brier = brier_score_loss(win_arr, propensity_win)
    prob_true, prob_pred = calibration_curve(win_arr, propensity_win, n_bins=10)
    calibration_error = float(np.mean(np.abs(prob_true - prob_pred)))

    result_win = WinPropensityResult(
        propensity=propensity_win,
        propensity_clipped=propensity_clipped,
        weights=weights,
        auc=auc_lgb,
        brier_score=brier,
        calibration_error=calibration_error,
    )

    # Feature importance
    importance_win = pd.DataFrame({
        "feature": feature_names,
        "importance": booster.feature_importance(),
    }).sort_values("importance", ascending=False)

    # --- LR (optional) ---
    propensity_lr: Optional[np.ndarray] = None
    auc_lr: Optional[float] = None

    lr_path = model_dir / "lr_win.joblib"
    if lr_path.exists():
        import joblib
        lr_artifact = joblib.load(str(lr_path))
        lr_model = lr_artifact["model"]
        scaler = lr_artifact["scaler"]
        lr_feature_names = lr_artifact["feature_names"]

        X_lr = df[lr_feature_names].fillna(0).values.astype(np.float32)
        X_lr_scaled = scaler.transform(X_lr)
        propensity_lr = lr_model.predict_proba(X_lr_scaled)[:, 1]
        auc_lr = roc_auc_score(win_arr, propensity_lr)

    return WinPropensityLoadedResult(
        result_win=result_win,
        importance_win=importance_win,
        propensity_lr=propensity_lr,
        auc_lr=auc_lr,
    )


def load_click_propensity_models(
    model_dir: Path,
    df: pd.DataFrame,
    clip_range: Tuple[float, float] = (0.001, 0.999),
) -> ClickPropensityLoadedResult:
    """Load pre-trained Click PS models from CLI and predict on df.

    Loads LGB Booster from ``lgb_ctr.txt`` and optionally LR from
    ``lr_ctr.joblib``.  Returns propensity scores, metrics, and
    feature importance — matching the interface previously produced
    by in-notebook training.

    Args:
        model_dir: Directory containing trained model files.
        df: DataFrame of **winners only** with feature columns and 'click' label.
        clip_range: Propensity clipping range for IPW weights.

    Returns:
        ClickPropensityLoadedResult with LGB (and optionally LR) outputs.
    """
    model_dir = Path(model_dir)

    # --- LGB Booster ---
    lgb_path = model_dir / "lgb_ctr.txt"
    if not lgb_path.exists():
        raise FileNotFoundError(
            f"LGB ctr model not found: {lgb_path}. "
            "Run: python scripts/train.py baseline --task ctr"
        )

    if not LGB_AVAILABLE:
        raise ImportError("lightgbm is required to load the LGB model")

    booster = lgb.Booster(model_file=str(lgb_path))
    feature_names = booster.feature_name()

    # Predict
    X_lgb = df[feature_names]
    propensity_click = booster.predict(X_lgb)

    # Clip + weights
    click_arr = df["click"].values
    propensity_clipped = np.clip(propensity_click, clip_range[0], clip_range[1])
    weights = click_arr.astype(float) / propensity_clipped

    # Metrics
    auc_lgb = roc_auc_score(click_arr, propensity_click)
    brier = brier_score_loss(click_arr, propensity_click)
    prob_true, prob_pred = calibration_curve(click_arr, propensity_click, n_bins=10)
    calibration_error = float(np.mean(np.abs(prob_true - prob_pred)))

    result_click = WinPropensityResult(
        propensity=propensity_click,
        propensity_clipped=propensity_clipped,
        weights=weights,
        auc=auc_lgb,
        brier_score=brier,
        calibration_error=calibration_error,
    )

    # Feature importance
    importance_click = pd.DataFrame({
        "feature": feature_names,
        "importance": booster.feature_importance(),
    }).sort_values("importance", ascending=False)

    # --- LR (optional) ---
    propensity_lr: Optional[np.ndarray] = None
    auc_lr: Optional[float] = None

    lr_path = model_dir / "lr_ctr.joblib"
    if lr_path.exists():
        import joblib
        lr_artifact = joblib.load(str(lr_path))
        lr_model = lr_artifact["model"]
        scaler = lr_artifact["scaler"]
        lr_feature_names = lr_artifact["feature_names"]

        X_lr = df[lr_feature_names].fillna(0).values.astype(np.float32)
        X_lr_scaled = scaler.transform(X_lr)
        propensity_lr = lr_model.predict_proba(X_lr_scaled)[:, 1]
        auc_lr = roc_auc_score(click_arr, propensity_lr)

    return ClickPropensityLoadedResult(
        result_click=result_click,
        importance_click=importance_click,
        propensity_lr=propensity_lr,
        auc_lr=auc_lr,
    )


def diagnose_win_propensity(
    propensity: np.ndarray,
    win: np.ndarray,
) -> Dict[str, Any]:
    """Diagnose win propensity for positivity violations.
    
    Args:
        propensity: Win propensity P(Win|X, bid)
        win: Win indicator
    
    Returns:
        Dictionary with diagnostic metrics
    """
    # AUC (lower is better for overlap)
    auc = roc_auc_score(win, propensity)
    
    # Overlap statistics
    overlap_01_09 = np.mean((propensity > 0.1) & (propensity < 0.9))
    overlap_005_095 = np.mean((propensity > 0.05) & (propensity < 0.95))
    
    # Extreme propensities
    extreme_low = np.mean(propensity < 0.01)
    extreme_high = np.mean(propensity > 0.99)
    
    # ESS
    weights = win.astype(float) / np.clip(propensity, 0.01, 0.99)
    ess = (weights.sum() ** 2) / (weights ** 2).sum()
    ess_ratio = ess / len(weights)
    
    return {
        "auc": auc,
        "overlap_01_09": overlap_01_09,
        "overlap_005_095": overlap_005_095,
        "extreme_low": extreme_low,
        "extreme_high": extreme_high,
        "ess": ess,
        "ess_ratio": ess_ratio,
        "positivity_violation": auc > 0.9,  # Warning threshold
    }
