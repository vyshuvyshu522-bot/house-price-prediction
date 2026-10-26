import pytest

from src.validation import validate_input

ZIPS = {"98103", "98004"}


def test_valid_input_is_converted_to_typed_values(valid_input):
    clean, errors = validate_input(valid_input, ZIPS)
    assert errors == {}
    assert clean["bedrooms"] == 3 and isinstance(clean["bedrooms"], int)
    assert clean["bathrooms"] == 2.0
    assert clean["zipcode"] == "98103"


def test_missing_fields_are_reported(valid_input):
    del valid_input["sqft_living"]
    valid_input["zipcode"] = "   "
    _, errors = validate_input(valid_input, ZIPS)
    assert set(errors) == {"sqft_living", "zipcode"}


@pytest.mark.parametrize("field,value", [
    ("sqft_living", "abc"), ("sqft_living", "nan"), ("sqft_living", "inf"),
    ("sqft_living", "-100"), ("sqft_living", "99999"),
    ("bedrooms", "2.5"), ("bedrooms", "0"), ("bedrooms", "50"),
    ("waterfront", "2"), ("grade", "14"), ("yr_built", "1800"), ("zipcode", "12345"),
])
def test_invalid_values_are_rejected(valid_input, field, value):
    valid_input[field] = value
    _, errors = validate_input(valid_input, ZIPS)
    assert field in errors


def test_extra_fields_are_ignored(valid_input):
    valid_input["unexpected"] = "x"
    clean, errors = validate_input(valid_input, ZIPS)
    assert errors == {} and "unexpected" not in clean
