"""Pipeline construction, cross-validation and evaluation."""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numeric columns, impute + one-hot encode ZIP code.

    Every step is learned inside the pipeline, so when the pipeline is fit on
    the training fold only, nothing from the test data leaks in.
    """
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),  # unseen ZIPs -> all zeros
    ])
    return ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])


def build_pipeline(regressor) -> Pipeline:
    """Preprocessing + regressor. Prices are right-skewed, so the model is
    trained on log(1 + price) and predictions are converted back to dollars."""
    target_wrapped = TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([("preprocess", build_preprocessor()), ("model", target_wrapped)])


def candidate_models(seed=config.RANDOM_STATE) -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=seed
        ),
    }


def regression_metrics(y_true, y_pred) -> dict:
    """All metrics are in dollars (predictions are already back-transformed)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        # Median absolute percentage error: a "typical %" miss, robust to outliers.
        "mdape": float(np.median(np.abs(y_pred - y_true) / y_true) * 100),
    }


def cross_validate_model(regressor, X, y, groups, folds=config.CV_FOLDS) -> dict:
    """Grouped K-fold CV on the TRAINING set. Returns mean and std per metric."""
    per_fold = []
    for train_idx, val_idx in GroupKFold(n_splits=folds).split(X, y, groups):
        pipe = build_pipeline(regressor).fit(X.iloc[train_idx], y.iloc[train_idx])
        per_fold.append(regression_metrics(y.iloc[val_idx], pipe.predict(X.iloc[val_idx])))
    return {
        name: {"mean": float(np.mean([f[name] for f in per_fold])),
               "std": float(np.std([f[name] for f in per_fold]))}
        for name in per_fold[0]
    }
