import jsonschema
import json
import re

from src.env.nutrition.schema import nutrition_schema, workout_one_week_schema, daily_meal_plan_schema

from jsonschema import validate, ValidationError


# Utils
def _extract_first_json_segment(s: str) -> str | None:
    start_candidates = [s.find('{'), s.find('[')]
    start_candidates = [i for i in start_candidates if i != -1]
    if not start_candidates: return None
    start = min(start_candidates)
    depth_obj, depth_arr, in_str, esc = 0, 0, False, False
    opener = s[start]
    want_obj, want_arr = opener == '{', opener == '['
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == '{': depth_obj += 1
            elif ch == '}': depth_obj -= 1
            elif ch == '[': depth_arr += 1
            elif ch == ']': depth_arr -= 1
            if want_obj and depth_obj == 0 and depth_arr == 0 and i >= start: return s[start:i+1]
            if want_arr and depth_arr == 0 and depth_obj == 0 and i >= start: return s[start:i+1]
    return None

def get_payload(obj):
    if hasattr(obj, "answer"): obj = obj.answer
    if isinstance(obj, str):
        try: obj = json.loads(obj)
        except:
             seg = _extract_first_json_segment(obj)
             if seg: 
                 try: obj = json.loads(seg)
                 except: pass
    if isinstance(obj, dict): return obj
    if isinstance(obj, list): return {"_root": obj}
    return {}

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

def verify_nutrition_schema(plan_json):
    """
    Returns (bool, message) whether the JSON matches the schema.
    """
    
    
    
    try:
        validate(instance=plan_json, schema=nutrition_schema)
        verification = True
        return float(verification), "schema OK"
    except ValidationError as e:
        return -1.0, f"schema error: {e.message}"

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

def verify_macros(plan_json, daily_cal_target, daily_prot_target, tol_pct=0.05):
    """
    Checks each day’s summed calories and proteins are within ± tol_pct.
    Returns list of error messages (empty = pass).
    """
    errors = []
    for day in plan_json.get("dailyMealPlans", []):
        d = day["day"]
        meals = day.get("meals", [])
        tot_c = sum(m.get("calories", 0) for m in meals)
        tot_p = sum(m.get("proteins", 0) for m in meals)
        low_c = daily_cal_target * (1 - tol_pct)
        high_c = daily_cal_target * (1 + tol_pct)
        if not (low_c <= tot_c <= high_c):
            errors.append(f"Day {d} calories {tot_c:.1f} outside [{low_c:.1f}, {high_c:.1f}]")
        low_p = daily_prot_target * (1 - tol_pct)
        high_p = daily_prot_target * (1 + tol_pct)
        if not (low_p <= tot_p <= high_p):
            errors.append(f"Day {d} protein {tot_p:.1f} outside [{low_p:.1f}, {high_p:.1f}]")
    return errors

def verify_daily_meal_plan_macros(plan_json, daily_cal_target, daily_prot_target, tol_pct=0.05):
    """
    Checks each day’s summed calories and proteins are within ± tol_pct.
    Returns list of error messages (empty = pass).
    """
    errors = []
    daily_meal_plan = plan_json.get("meals", [])
    tot_c = 0
    tot_p = 0
    low_c = daily_cal_target * (1 - tol_pct)
    high_c = daily_cal_target * (1 + tol_pct)
    low_p = daily_prot_target * (1 - tol_pct)
    high_p = daily_prot_target * (1 + tol_pct)
    for meal in daily_meal_plan:
        c = meal.get("calories", 0)
        p = meal.get("proteins", 0)
        
        tot_c += c
        tot_p += p


    if not (low_c <= tot_c <= high_c):
            errors.append(f"Calories {tot_c:.1f} outside [{low_c:.1f}, {high_c:.1f}]")
    if not (low_p <= tot_p <= high_p):
        errors.append(f"Protein {tot_p:.1f} outside [{low_p:.1f}, {high_p:.1f}]")
    

    print(tot_c, tot_p)
    if errors:
        return -1.0, {"macro_errors": errors}
    return 1.0, {"ok": True}

def verify_no_banned(plan_json, banned_keywords):
    """
    Ensure no banned keywords appear in meal name or description.
    Returns list of (day, meal_name, offending_keyword) if violations.
    """
    violations = []
    for day in plan_json.get("dailyMealPlans", []):
        d = day.get("day")
        for m in day.get("meals", []):
            text = (m.get("name", "") + " " + m.get("description", "")).lower()
            for kw in banned_keywords:
                if re.search(rf"\b{re.escape(kw.lower())}\b", text):
                    violations.append((d, m.get("name", ""), kw))
    return violations

def verify_workout_week(week_plan):
    """
    week_plan: list of day dicts (from plan_json["workouts"])
    Returns list of error messages.
    """
    errors = []
    seen_days = set()
    rest_days = 0
    for entry in week_plan:
        d = entry.get("day")
        if d is None or not (0 <= d <= 6):
            errors.append(f"Invalid day index {d}")
        else:
            if d in seen_days:
                errors.append(f"Duplicate day {d}")
            seen_days.add(d)
        exs = entry.get("exercises", [])
        if not exs:
            rest_days += 1
        else:
            for e in exs:
                if ("sets" not in e) or ("reps" not in e) or ("restTime" not in e):
                    errors.append(f"Day {d} exercise missing key: {e.get('exercise')}")
    # Check coverage: you expect one entry per day 0..6 or at least unique set
    if len(seen_days) != len(week_plan):
        errors.append("Mismatch: workout list length vs unique days")
    if rest_days < 1:
        errors.append("Less than 1 rest day")
    return errors



# --- Combined verifier APIs ---

def verify_nutrition_plan(plan_json, daily_cal_target, daily_prot_target, banned_keywords=None):
    """
    Returns (score, diagnostic_info)
    score is in [0,1]. 1 means fully valid.
    """
    ok, msg = verify_nutrition_schema(plan_json)
    if not ok:
        return -1.0, {"schema_error": msg}

    macro_errs = verify_macros(plan_json, daily_cal_target, daily_prot_target)
    if macro_errs:
        return -1.0, {"macro_errors": macro_errs}

    if banned_keywords:
        banned_viol = verify_no_banned(plan_json, banned_keywords)
        if banned_viol:
            return -1.0, {"banned_violations": banned_viol}

    return 1.0, {"ok": True}

def verify_workout_plan(plan_json):
    """
    plan_json: dict with key "workouts"
    Returns (score, diagnostics)
    """
    ok, msg = verify_workout_schema(plan_json)
    if not ok:
        return -1.0, {"schema_error": msg}

    week = plan_json.get("workouts", [])
    errs = verify_workout_week(week)
    if errs:
        return -1.0, {"week_errors": errs}

    return 1.0, {"ok": True}

