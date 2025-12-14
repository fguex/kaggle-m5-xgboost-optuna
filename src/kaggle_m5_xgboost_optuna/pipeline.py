"""Pipeline module exposing TrendResidualsEnsemble.

This module provides a small sklearn-like estimator that fits a LinearRegression
trend model and an XGBoost regressor on residuals. It's safe for notebook use
and performs conversions to numpy.float32 internally.

Saved models are meant to be written under the package's `models/` directory,
e.g. `src/kaggle_m5_xgboost_optuna/models/`.
"""
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
import os
import sys
from typing import Optional, Tuple, Any
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


class TrendResidualsEnsemble(BaseEstimator, RegressorMixin):
    """Two-stage estimator: LinearRegression trend + XGBoost on residuals.

    Methods:
    - fit(X, y, eval_set=None): fits linear trend then XGBRegressor on residuals
    - predict(X): returns trend_pred + xgb_pred
    - save(path): persist the pipeline to a file using joblib
    - load(path): classmethod to load a saved pipeline
    """

    def __init__(self, xgb_params: Optional[dict] = None, scaler: Optional[Any] = None):
        self.trend_model = LinearRegression()
        self.xgb_params = xgb_params or {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'tree_method': 'hist',
        }
        # xgb_model will hold the XGBRegressor instance after fit
        self.xgb_model = None
        # Optional feature scaler (None => no scaling). If you pass the string
        # 'standard' the pipeline will create a StandardScaler instance. You
        # can also pass any sklearn-like scaler instance (must implement
        # fit/transform). The scaler is fit on training features and applied
        # to validation/test features during predict. The fitted scaler object
        # is stored in `_scaler_obj` for use at predict/save/load time.
        self.scaler = scaler
        self._scaler_obj = None

    def _to_numpy(self, X):
        if hasattr(X, 'to_numpy'):
            return X.to_numpy().astype(np.float32)
        else:
            return np.asarray(X, dtype=np.float32)

    def _to_1d(self, y):
        if hasattr(y, 'to_numpy'):
            return y.to_numpy().ravel().astype(np.float32)
        else:
            return np.ravel(y).astype(np.float32)

    @staticmethod
    def _wrmse(y_true, y_pred):
        """Compute weighted RMSE with weights 1..n (matches WRMSE/WRMSEE used in the notebook)."""
        yt = np.asarray(y_true).ravel()
        yp = np.asarray(y_pred).ravel()
        if len(yt) != len(yp) or len(yt) == 0:
            raise ValueError('y_true and y_pred must have the same non-zero length')
        weights = np.arange(1, len(yt) + 1)
        return float(np.sqrt(mean_squared_error(yt, yp, sample_weight=weights)))

    def fit(self, X, y, eval_set: Optional[Tuple] = None):
        # Resolve xgboost module at runtime. Prefer an already-imported xgboost from sys.modules
        # (e.g. the user executed `import xgboost as xgb` in the notebook). Fall back to importing.
        xgb_mod = sys.modules.get('xgboost')
        if xgb_mod is None:
            try:
                import xgboost as xgb_mod  # lazy import
            except Exception as e:
                raise ImportError('xgboost is required to fit the residual model; please install xgboost') from e
        X_np = self._to_numpy(X)
        # Fit scaler on training features if requested
        if self.scaler is not None:
            if isinstance(self.scaler, str) and self.scaler.lower() == 'standard':
                self._scaler_obj = StandardScaler().fit(X_np)
            elif hasattr(self.scaler, 'fit') and hasattr(self.scaler, 'transform'):
                # user provided scaler instance
                self._scaler_obj = self.scaler.fit(X_np)
            else:
                raise ValueError("scaler must be None, 'standard', or an sklearn-like scaler instance")
            X_np = self._scaler_obj.transform(X_np)
        y_np = self._to_1d(y)

        # Fit linear trend
        self.trend_model = LinearRegression()
        self.trend_model.fit(X_np, y_np)
        trend_pred = self.trend_model.predict(X_np).astype(np.float32)

        # Residuals
        residuals = (y_np - trend_pred).astype(np.float32)

        # Fit XGBRegressor on residuals using the xgboost module imported at top
        params = dict(self.xgb_params)
        params.setdefault('tree_method', 'hist')
        params.setdefault('random_state', 19970507)
        # Remove any user-provided 'eval_metric' so it doesn't override
        # our custom WRMSE evaluation during fit. If the user passed
        # eval_metric='rmse' (or similar) in xgb_params, that would cause
        # validation logs to show RMSE instead of WRMSE even when we pass
        # a callable eval_metric to fit(). Pop it here to ensure our
        # _wrmse_eval is the authoritative metric used for monitoring.
        params.pop('eval_metric', None)

        # XGBRegressor expects n_estimators; pop if provided
        n_estimators = params.pop('n_estimators', None)
        # use xgb_mod.XGBRegressor (resolved above)
        self.xgb_model = xgb_mod.XGBRegressor(**params, n_estimators=n_estimators or 100)

        # custom eval metric for WRMSE to align training/early-stopping with competition metric
        def _wrmse_eval(preds, dmatrix):
            labels = dmatrix.get_label()
            preds = np.asarray(preds).ravel()
            labels = np.asarray(labels).ravel()
            if len(labels) == 0:
                return 'wrmse', float('nan')
            w = np.arange(1, len(labels) + 1).astype(float)
            val = float(np.sqrt(np.sum(w * (labels - preds) ** 2) / np.sum(w)))
            return 'wrmse', val

        # Fit with support for multiple xgboost versions:
        if eval_set is not None:
            Xv, yv = eval_set
            Xv_np = self._to_numpy(Xv)
            if self._scaler_obj is not None:
                Xv_np = self._scaler_obj.transform(Xv_np)
            yv_np = self._to_1d(yv)

            # IMPORTANT: compute validation residuals (y_val - trend_pred) so XGB models the
            # residuals and early stopping/monitoring is done on the residual target.
            trend_val_pred = self.trend_model.predict(Xv_np).astype(np.float32)
            val_residuals = (yv_np - trend_val_pred).astype(np.float32)

            # Try several fit signatures to support different xgboost versions
            try:
                # Preferred simple call (works on many xgboost versions)
                self.xgb_model.fit(
                    X_np,
                    residuals,
                    eval_set=[(Xv_np, val_residuals)],
                    early_stopping_rounds=50,
                    verbose=False,
                    eval_metric=_wrmse_eval,
                )
            except Exception:
                # Some xgboost versions' sklearn wrapper don't accept early_stopping_rounds kwarg.
                # Try using the callback API for early stopping first.
                tried = False
                try:
                    cb = xgb_mod.callback.EarlyStopping(rounds=50, save_best=True)
                    self.xgb_model.fit(
                        X_np,
                        residuals,
                        eval_set=[(Xv_np, val_residuals)],
                        callbacks=[cb],
                        verbose=False,
                        eval_metric=_wrmse_eval,
                    )
                    tried = True
                except Exception:
                    # Next fallback: try fit with eval_set but without early stopping/verbose
                    try:
                        self.xgb_model.fit(X_np, residuals, eval_set=[(Xv_np, val_residuals)], eval_metric=_wrmse_eval)
                        tried = True
                    except Exception:
                        # Final fallback: fit without eval_set (best-effort)
                        try:
                            self.xgb_model.fit(X_np, residuals)
                            tried = True
                        except Exception as final_e:
                            # If none of the above succeeded, raise a clear error
                            raise RuntimeError('Failed to fit XGBRegressor with available API signatures') from final_e
        else:
            # No eval set provided; simple fit
            # when no eval_set is provided, still pass the custom eval metric for consistency
            try:
                self.xgb_model.fit(X_np, residuals, verbose=False, eval_metric=_wrmse_eval)
            except Exception:
                # fallback to plain fit
                self.xgb_model.fit(X_np, residuals, verbose=False)

        return self

    def predict(self, X):
        X_np = self._to_numpy(X)
        trend_pred = self.trend_model.predict(X_np)
        # If residual model wasn't trained (None), return trend predictions only
        if self.xgb_model is None:
            return trend_pred
        xgb_pred = self.xgb_model.predict(X_np)
        return trend_pred + xgb_pred

    def score_wrmse(self, X, y):
        """Compute WRMSE on X, y using the ensemble's predictions.

        Args:
            X: features (pandas/array/polars)
            y: true target (pandas/array/polars)

        Returns:
            float: WRMSE value
        """
        if hasattr(y, 'to_numpy'):
            y_true = y.to_numpy().ravel()
        else:
            y_true = np.ravel(y)
        y_pred = self.predict(X)
        return self._wrmse(y_true, y_pred)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)
