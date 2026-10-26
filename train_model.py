"""Train, compare and save the house price model.

Usage:  python train_model.py
Everything is seeded, so repeated runs give the same numbers (same library versions).
"""
import argparse
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.inspection import permutation_importance

from src import config
from src.charts import plot_actual_vs_predicted, plot_feature_importance, plot_price_distribution
from src.data import clean_data, load_raw, split_train_test
from src.modeling import build_pipeline, candidate_models, cross_validate_model, regression_metrics


def main(data_path=config.DATA_PATH, seed=config.RANDOM_STATE):
    started = time.time()
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load + clean
    df, report = clean_data(load_raw(data_path))
    print(f"Rows: raw={report['rows_raw']}  clean={report['rows_clean']}")
    for key, value in report.items():
        if key.endswith("_removed"):
            print(f"  {key}: {value}")

    # 2. Split BEFORE any fitting. Group by house id so repeat sales never straddle the split.
    train, test = split_train_test(df, seed=seed)
    X_train, y_train = train[config.FEATURES], train[config.TARGET]
    X_test, y_test = test[config.FEATURES], test[config.TARGET]
    print(f"Train={len(train)}  Test={len(test)}")

    # 3. Cross-validate each candidate on the training set only -> pick by validation RMSE.
    models = candidate_models(seed)
    cv_results = {}
    for name, regressor in models.items():
        print(f"Cross-validating {name} ...")
        cv_results[name] = cross_validate_model(regressor, X_train, y_train, train[config.GROUP_COLUMN])
    selected = min(cv_results, key=lambda n: cv_results[n]["rmse"]["mean"])
    print(f"Selected by CV RMSE: {selected}")

    # 4. Refit every candidate on the full training set and score once on the untouched test set.
    fitted, test_results = {}, {}
    for name, regressor in models.items():
        fitted[name] = build_pipeline(regressor).fit(X_train, y_train)
        test_results[name] = regression_metrics(y_test, fitted[name].predict(X_test))

    best = fitted[selected]
    y_pred = best.predict(X_test)

    # 5. Permutation importance on the ORIGINAL columns (works for any model).
    perm = permutation_importance(best, X_test, y_test, n_repeats=5, random_state=seed,
                                  scoring="r2", n_jobs=1)
    importances = {f: float(v) for f, v in zip(config.FEATURES, perm.importances_mean)}

    # 6. Charts + artifacts
    plot_actual_vs_predicted(y_test, y_pred, selected, config.CHARTS_DIR / "actual_vs_predicted.png")
    plot_price_distribution(df[config.TARGET], config.CHARTS_DIR / "price_distribution.png")
    plot_feature_importance(importances, selected, config.CHARTS_DIR / "feature_importance.png")

    bundle = {
        "pipeline": best,
        "model_name": selected,
        "features": config.FEATURES,
        "zipcodes": sorted(df["zipcode"].dropna().unique().tolist()),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, config.MODEL_PATH, compress=3)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_model": selected,
        "selection_rule": f"lowest mean RMSE across {config.CV_FOLDS}-fold grouped cross-validation on the training set",
        "cv_metrics": cv_results,
        "test_metrics": test_results,
        "dataset": {**report, "train_rows": len(train), "test_rows": len(test),
                    "median_price": float(df[config.TARGET].median()),
                    "min_price": float(df[config.TARGET].min()),
                    "max_price": float(df[config.TARGET].max())},
        "feature_importance": dict(sorted(importances.items(), key=lambda kv: -kv[1])),
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "numpy": np.__version__, "random_state": seed},
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("\nCross-validation (mean over folds):")
    for name, res in cv_results.items():
        print(f"  {name:18s} MAE=${res['mae']['mean']:,.0f}  RMSE=${res['rmse']['mean']:,.0f}  R2={res['r2']['mean']:.3f}")
    print("Held-out test set:")
    for name, res in test_results.items():
        print(f"  {name:18s} MAE=${res['mae']:,.0f}  RMSE=${res['rmse']:,.0f}  R2={res['r2']:.3f}  MdAPE={res['mdape']:.1f}%")
    size_mb = config.MODEL_PATH.stat().st_size / 1e6
    print(f"\nSaved {config.MODEL_PATH.name} ({size_mb:.1f} MB) and metrics.json in {time.time() - started:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=config.DATA_PATH, help="path to the CSV dataset")
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    args = parser.parse_args()
    main(args.data, args.seed)
