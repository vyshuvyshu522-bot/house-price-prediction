"""Flask application: home/prediction, dashboard and prediction history."""
import json
import logging
import os

from flask import Flask, jsonify, redirect, render_template, request, url_for

from src import config, database
from src.config import FEATURE_SPECS, FEATURES
from src.predictor import ModelNotAvailableError, Predictor
from src.validation import validate_input


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.update(
        DATABASE=str(config.DATABASE_PATH),
        MODEL_PATH=str(config.MODEL_PATH),
        METRICS_PATH=str(config.METRICS_PATH),
    )
    if test_config:
        app.config.update(test_config)
    logging.basicConfig(level=logging.INFO)

    database.init_db(app.config["DATABASE"])

    # The app still starts if the model is missing; pages then explain what to do.
    predictor, model_error = None, None
    try:
        predictor = Predictor.load(app.config["MODEL_PATH"])
    except ModelNotAvailableError as exc:
        model_error = str(exc)
        app.logger.warning("Model not loaded: %s", exc)

    def load_metrics():
        try:
            with open(app.config["METRICS_PATH"]) as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

    @app.template_filter("money")
    def money(value):
        return f"${value:,.0f}"

    @app.context_processor
    def inject_globals():
        return {"specs": FEATURE_SPECS, "model_ready": predictor is not None}

    def render_home(form_values=None, errors=None, result=None, status=200):
        metrics = load_metrics()
        typical = None
        if metrics and predictor:
            test = metrics["test_metrics"].get(predictor.model_name)
            if test:
                typical = {"mdape": test["mdape"], "mae": test["mae"]}
        defaults = {n: s.get("default") for n, s in FEATURE_SPECS.items() if "default" in s}
        return render_template(
            "index.html",
            zipcodes=sorted(predictor.zipcodes) if predictor else [],
            values=form_values if form_values is not None else defaults,
            errors=errors or {}, result=result, model_error=model_error, typical=typical,
        ), status

    @app.get("/")
    def home():
        return render_home()

    @app.post("/predict")
    def predict():
        wants_json = request.is_json or request.accept_mimetypes.best == "application/json"
        raw = request.get_json(silent=True) if request.is_json else request.form.to_dict()
        if not isinstance(raw, dict):
            raw = {}

        if predictor is None:
            message = model_error or "The prediction model is not available."
            if wants_json:
                return jsonify(ok=False, message=message), 503
            return render_home(form_values=raw, status=503)

        values, errors = validate_input(raw, predictor.zipcodes)
        if errors:
            if wants_json:
                return jsonify(ok=False, message="Please fix the highlighted fields.", errors=errors), 400
            return render_home(form_values=raw, errors=errors, status=400)

        try:
            price = predictor.predict(values)
            database.save_prediction(app.config["DATABASE"], values, price, predictor.model_name)
        except Exception:  # never leak a stack trace to the user
            app.logger.exception("Prediction failed")
            message = "Something went wrong while estimating the price. Please try again."
            if wants_json:
                return jsonify(ok=False, message=message), 500
            return render_home(form_values=raw, errors={"_form": message}, status=500)

        if wants_json:
            return jsonify(ok=True, price=price, price_formatted=f"${price:,.0f}", model=predictor.model_name)
        return render_home(form_values=raw, result={"price": price, "model": predictor.model_name})

    @app.get("/dashboard")
    def dashboard():
        metrics = load_metrics()
        charts_ready = all((config.CHARTS_DIR / f).exists() for f in
                           ("actual_vs_predicted.png", "price_distribution.png", "feature_importance.png"))
        return render_template("dashboard.html", metrics=metrics, charts_ready=charts_ready)

    @app.get("/history")
    def history():
        try:
            rows = database.get_history(app.config["DATABASE"])
            error = None
        except Exception:
            app.logger.exception("Could not read history")
            rows, error = [], "Prediction history could not be loaded."
        return render_template("history.html", rows=rows, features=FEATURES, error=error)

    @app.post("/history/clear")
    def clear_history():
        database.clear_history(app.config["DATABASE"])
        return redirect(url_for("history"))

    @app.errorhandler(404)
    def not_found(_):
        return render_template("error.html", code=404, message="That page does not exist."), 404

    @app.errorhandler(405)
    def not_allowed(_):
        return render_template("error.html", code=405, message="That action is not allowed here."), 405

    @app.errorhandler(500)
    def server_error(_):
        return render_template("error.html", code=500, message="Something went wrong on our side."), 500

    return app


if __name__ == "__main__":
    # debug stays off by default so the Werkzeug debugger (which shows stack traces) is never exposed.
    create_app().run(debug=os.environ.get("FLASK_DEBUG") == "1")
