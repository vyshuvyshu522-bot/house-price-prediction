"""Server-side validation of prediction form / JSON input."""
import math

from src.config import FEATURE_SPECS, FEATURES


def _is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def validate_input(raw: dict, valid_zipcodes=None):
    """Return (clean_values, errors).

    clean_values holds typed values ready for the model; errors maps
    field name -> human-readable message. Unknown extra fields are ignored.
    """
    clean, errors = {}, {}
    for name in FEATURES:
        spec = FEATURE_SPECS[name]
        value = raw.get(name)
        label = spec["label"]

        if _is_blank(value):
            errors[name] = f"{label} is required."
            continue

        if spec["kind"] == "choice":  # ZIP code
            code = str(value).strip()
            if valid_zipcodes is not None and code not in valid_zipcodes:
                errors[name] = "Choose a ZIP code from the list (the model only knows King County, WA ZIP codes)."
            else:
                clean[name] = code
            continue

        try:
            number = float(str(value).strip())
        except (TypeError, ValueError):
            errors[name] = f"{label} must be a number."
            continue
        if not math.isfinite(number):  # rejects "nan" / "inf"
            errors[name] = f"{label} must be a finite number."
            continue
        if spec["kind"] == "int":
            if not number.is_integer():
                errors[name] = f"{label} must be a whole number."
                continue
            number = int(number)
        if not spec["min"] <= number <= spec["max"]:
            errors[name] = f"{label} must be between {spec['min']:,} and {spec['max']:,}."
            continue
        clean[name] = number
    return clean, errors
