from src import database


def test_save_and_read_history(tmp_path, valid_input):
    db = tmp_path / "h.db"
    database.init_db(db)
    values = {**valid_input, "zipcode": "98103"}
    database.save_prediction(db, values, 450000, "Random Forest")
    database.save_prediction(db, values, 460000, "Random Forest")
    rows = database.get_history(db)
    assert [r["predicted_price"] for r in rows] == [460000, 450000]   # newest first
    assert rows[0]["zipcode"] == "98103"


def test_clear_history(tmp_path, valid_input):
    db = tmp_path / "h.db"
    database.init_db(db)
    database.save_prediction(db, valid_input, 1, "m")
    database.clear_history(db)
    assert database.get_history(db) == []
