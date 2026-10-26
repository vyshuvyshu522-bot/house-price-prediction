# House Price Prediction Using Machine Learning

A complete machine-learning web application: it trains Linear Regression and Random Forest models on real
house-sale data, compares them with proper validation, serves predictions through a Flask web app, shows a
results dashboard, and stores every prediction in SQLite.

## Problem statement

Given a few basic characteristics of a house (living area, bedrooms, bathrooms, build grade, ZIP code, ...),
estimate its sale price. The project focuses on doing the machine-learning workflow *correctly*: no data
leakage, model choice by validation performance, honest reporting, and a tested application around the model.

## Features

- Responsive home page with a validated prediction form (client-side checks + server-side validation)
- Linear Regression vs Random Forest, compared with MAE, RMSE and R²
- Results dashboard: metrics table, actual-vs-predicted chart, price distribution, permutation feature importance
- Prediction history stored in SQLite, with a page to review (and clear) it
- Friendly error handling: invalid input, missing fields, missing/corrupt model; no stack traces shown to users
- 39 automated tests (pytest)

## Technology stack

Python 3.11+, Pandas, NumPy, scikit-learn, Flask, SQLite (`sqlite3`), Matplotlib (chart images),
HTML / CSS / vanilla JavaScript, pytest.

## Dataset

**House Sales in King County, USA** - 21,613 sales of homes in King County, Washington (Seattle area),
May 2014 - May 2015. Originally published on Kaggle
(<https://www.kaggle.com/datasets/harlfoxem/housesalesprediction>); the copy in `data/` was downloaded from a
public GitHub mirror. Details, checksum and a licensing note are in [`data/README.md`](data/README.md).

Cleaning (see `src/data.py`): 3 exact duplicate rows and 1 impossible 33-bedroom row removed (21,609 rows kept).
The dataset has no missing values, but the pipeline still imputes them (median / most frequent) so the app
and tests handle missing inputs safely.

## How the ML part works

1. **Split first.** 80/20 train/test split *grouped by house id* (`GroupShuffleSplit`), because 177 houses
   were sold twice; a plain random split would put the same house in both sets and inflate the scores.
2. **One pipeline.** Median-impute + standardise numeric features; most-frequent-impute + one-hot encode ZIP code
   (unseen ZIPs are ignored). Prices are right-skewed, so models predict `log(1 + price)` and are converted back to
   dollars. The preprocessing is fit only on training data.
3. **Select with validation, not the test set.** 5-fold grouped cross-validation on the training set;
   the model with the lowest mean RMSE is selected.
4. **Report once on the test set.** Both models are refit on all training data and scored on the untouched test set.

## Model evaluation results (actual output of `python train_model.py`)

Train rows: 17,288, test rows: 4,321, `random_state=42`. All errors are in US dollars.


| Model             | Split                  |     MAE |     RMSE |    R² |
| ----------------- | ---------------------- | ------: | -------: | ----: |
| Linear Regression | 5-fold CV (validation) | $80,140 | $200,988 | 0.685 |
| Random Forest     | 5-fold CV (validation) | $90,373 | $169,620 | 0.784 |
| Linear Regression | Test (held out)        | $78,085 | $144,288 | 0.850 |
| Random Forest     | Test (held out)        | $90,659 | $161,936 | 0.811 |

**Selected model: Random Forest** (lowest mean cross-validation RMSE).

Read these numbers carefully:

- The two models are close and the ranking is **not consistent** across metrics. Linear Regression has the lower
  MAE in both CV and test, and the better test RMSE and R². Random Forest won the selection only because of CV RMSE.
- The reason is visible in the cross-validation folds: Linear Regression (fit on log price) *extrapolates
  exponentially* for the largest houses. In one fold it predicted about $16M for a 13,540 sq ft house that sold for
  $2.28M, so its fold RMSE ranged from $165K to $272K (R² from 0.43 to 0.78). Random Forest was stable
  (CV RMSE std about $4K). The test set happened not to contain such a case.
- I kept the selection rule fixed in advance instead of switching to Linear Regression after seeing the test
  results, because choosing a model by its test score would make that test score optimistic.
- Median absolute percentage error on the test set: Random Forest 11.3%, Linear Regression 10.1%.
- No hyperparameter tuning was done; the Random Forest uses fixed, reasonable settings.

## Folder structure

```
house-price-prediction/
├── app.py                  # Flask app factory + routes
├── train_model.py          # reproducible training / evaluation script
├── requirements.txt
├── README.md
├── .gitignore
├── data/                   # kc_house_data.csv + data/README.md (source, checksum)
├── models/                 # generated: house_price_model.joblib (git-ignored), metrics.json
├── src/
│   ├── config.py           # paths, feature lists, form/validation specs
│   ├── data.py             # load, clean, grouped split
│   ├── modeling.py         # pipeline, cross-validation, metrics
│   ├── predictor.py        # model loading + prediction
│   ├── validation.py       # server-side input validation
│   ├── database.py         # SQLite history
│   └── charts.py           # dashboard chart images
├── templates/              # base, index, dashboard, history, error
├── static/{css,js,img}/    # styles, form script, generated charts
└── tests/                  # pytest suite
```

## Installation

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

This writes `models/house_price_model.joblib`, `models/metrics.json` and the three charts in `static/img/`.
It took about 2-3 minutes on a single CPU core. The model file is git-ignored; everyone regenerates it with this
command. (A saved model only loads with the same scikit-learn version, hence the pinned `requirements.txt`.)

## Run the app

```bash
flask --app app run
# or: python app.py
```

Open <http://127.0.0.1:5000>. Prediction history is stored in `instance/predictions.db` (created automatically).

## Run the tests

```bash
python -m pytest
```

Tests use a small model trained on a sample of the dataset, so they run in seconds and never touch your real model.
Actual result when this project was built: **39 passed**.

## Screenshots

Not committed yet. After running the app, capture these and save them in `docs/screenshots/`:
`home.png` (form + result), `dashboard.png`, `history.png`; then add them here with
`![Home](docs/screenshots/home.png)`.

## Limitations

- Trained on **King County, WA sales from 2014-2015**: it does not reflect current prices or other cities/countries.
- Only 11 input features; no exact location, interior quality, renovations or market trends.
- Predictions are estimates, not valuations. A typical error is about 10-11% of the price, and individual errors are
  much larger, especially for expensive homes (Random Forest tends to under-predict the top end; see the
  actual-vs-predicted chart).
- The history page is shared by everyone using the app (no user accounts), and the "clear history" form has no CSRF protection.
- Random Forest cannot predict above the prices it saw in training, and the form is restricted to the training data's ranges.

## Future improvements

Hyperparameter search; gradient boosting; prediction intervals (e.g. quantile regression); use latitude/longitude;
user accounts and CSRF protection; Docker image; deployment.
