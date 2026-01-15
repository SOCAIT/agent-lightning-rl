import jsonschema
import json
import re

from src.constants.schema import nutrition_schema, workout_one_week_schema, daily_meal_plan_schema

from jsonschema import validate, ValidationError

def is_valid_json(data):
    """
    Check if data is valid JSON (either already parsed dict or parseable string).
    Returns (bool, parsed_data_or_error_msg)
    """
    if isinstance(data, dict):
        return float(True), data
    
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
            return float(True), parsed
        except json.JSONDecodeError as e:
            return False, f"JSON parse error: {e.msg} at position {e.pos}"
    
    return float(False), f"Invalid input type: {type(data).__name__}"



def verify_sevendays_meal_plan_schema(plan_json):
    """
    Verifies that the plan has exactly 7 daily meal plans.
    Returns (score: float, diag: dict)
    """
    key = "dailyMealPlans"
    if key not in plan_json:
        return 0.0, {"error": "Missing 'dailyMealPlans' key"}

    daily_meals = plan_json[key]
    if not isinstance(daily_meals, (list, tuple)):
        return 0.0, {"error": "'dailyMealPlans' must be a list"}
    
    if len(daily_meals) != 7:
        return 0.0, {"error": f"'dailyMealPlans' length is {len(daily_meals)}, expected 7"}

    return 1.0, {"ok": True}


def verify_nutrition_schema(plan_json, nutrition_schema=nutrition_schema):
    """
    Verifies that the JSON matches both:
    1. The 7-day meal plan structure.
    2. The nutrition schema.
    Returns (score: float, diag: dict)
    """
    # Step 1: Check the 7-day structure
    score, diag = verify_sevendays_meal_plan_schema(plan_json)
    if score == 0.0:
        diag["error_type"] = "sevendays_meal_plan_schema"
        return 0.0, diag

    # Step 2: Validate against JSON schema
    try:
        validate(instance=plan_json, schema=nutrition_schema)
        return 1.0, {"ok": True}
    except ValidationError as e:
        path = " -> ".join(map(str, e.path)) if e.path else "(root)"
        return 0.0, {"error": f"Schema validation failed at {path}: {e.message}"}

def verify_meal_plan_schema(plan_json):
    """
    Returns (bool, message) whether the JSON matches the schema.
    """
    
    try:
        validate(instance=plan_json, schema=daily_meal_plan_schema)
        verification = True
        return float(verification), "schema OK"
    except ValidationError as e:
        return -1.0, f"schema error: {e.message}"


    
    


def verify_workout_schema(plan_json):
    """Check that the workout plan JSON matches 1-week workout schema."""
    
    
    try:
        validate(instance=plan_json, schema=workout_one_week_schema)
        verification = True
        return float(verification), "schema OK"
    except ValidationError as e:
        return -1.0, f"schema error: {e.message}"
