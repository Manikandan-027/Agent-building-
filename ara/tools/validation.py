"""Deterministic input/output validation for every tool call (jsonschema)."""
from __future__ import annotations

import jsonschema
from jsonschema import Draft202012Validator

from ara.core.errors import ValidationError


def validate_input(args: dict, schema: dict, tool_name: str) -> dict:
    """Validate against the tool's declared schema; reject unknown extra fields."""
    schema = {**schema, "additionalProperties": False}
    try:
        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(args), key=lambda e: e.json_path)
        if errors:
            details = [{"path": e.json_path, "msg": e.message} for e in errors[:5]]
            raise ValidationError(f"invalid arguments for tool '{tool_name}'", details={"errors": details})
    except jsonschema.SchemaError as exc:
        raise ValidationError(f"tool '{tool_name}' has a malformed schema: {exc.message}") from exc
    return args


def validate_output(result: object, schema: dict, tool_name: str) -> object:
    schema = {**schema, "additionalProperties": False} if schema.get("type") == "object" else schema
    try:
        jsonschema.validate(result, schema)
    except jsonschema.ValidationError as exc:
        raise ValidationError(
            f"tool '{tool_name}' returned schema-invalid output", details={"path": exc.json_path, "msg": exc.message}
        ) from exc
    except jsonschema.SchemaError as exc:
        raise ValidationError(f"tool '{tool_name}' has a malformed output schema: {exc.message}") from exc
    return result
