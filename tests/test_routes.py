from app import create_app
from src import database


def test_main_pages_load(client):
    for path in ("/", "/dashboard", "/history"):
        assert client.get(path).status_code == 200


def test_home_shows_form_fields(client):
    html = client.get("/").get_data(as_text=True)
    assert 'name="sqft_living"' in html and 'name="zipcode"' in html


def test_valid_form_prediction_is_saved(client, app, valid_input):
    response = client.post("/predict", data=valid_input)
    assert response.status_code == 200
    assert "Estimated price" in response.get_data(as_text=True)
    assert len(database.get_history(app.config["DATABASE"])) == 1
    assert "98103" in client.get("/history").get_data(as_text=True)


def test_valid_json_prediction(client, valid_input):
    response = client.post("/predict", data=valid_input, headers={"Accept": "application/json"})
    body = response.get_json()
    assert response.status_code == 200 and body["ok"] is True
    assert body["price"] > 0 and body["price_formatted"].startswith("$")


def test_invalid_input_returns_400_and_is_not_saved(client, app, valid_input):
    valid_input["sqft_living"] = "-5"
    response = client.post("/predict", data=valid_input, headers={"Accept": "application/json"})
    assert response.status_code == 400
    assert "sqft_living" in response.get_json()["errors"]
    assert database.get_history(app.config["DATABASE"]) == []


def test_missing_fields_show_errors_in_html(client, valid_input):
    del valid_input["bedrooms"]
    response = client.post("/predict", data=valid_input)
    assert response.status_code == 400
    assert "Bedrooms is required." in response.get_data(as_text=True)


def test_empty_post_and_bad_json_do_not_crash(client):
    assert client.post("/predict", data={}).status_code == 400
    assert client.post("/predict", data="[1,2]", content_type="application/json").status_code == 400


def test_app_works_without_model_and_hides_details(tmp_path, valid_input):
    app = create_app({"TESTING": True, "DATABASE": str(tmp_path / "t.db"),
                      "MODEL_PATH": str(tmp_path / "missing.joblib"),
                      "METRICS_PATH": str(tmp_path / "missing.json")})
    client = app.test_client()
    home = client.get("/")
    assert home.status_code == 200 and "not available" in home.get_data(as_text=True)
    response = client.post("/predict", data=valid_input, headers={"Accept": "application/json"})
    assert response.status_code == 503
    assert "Traceback" not in response.get_data(as_text=True)


def test_prediction_failure_returns_friendly_500(client, app, valid_input, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(database, "save_prediction", boom)
    response = client.post("/predict", data=valid_input, headers={"Accept": "application/json"})
    assert response.status_code == 500
    text = response.get_data(as_text=True)
    assert "secret internal detail" not in text and "Traceback" not in text


def test_unknown_page_and_wrong_method(client):
    assert client.get("/nope").status_code == 404
    assert client.get("/predict").status_code == 405


def test_clear_history(client, valid_input):
    client.post("/predict", data=valid_input)
    response = client.post("/history/clear", follow_redirects=True)
    assert response.status_code == 200 and "No predictions yet" in response.get_data(as_text=True)
