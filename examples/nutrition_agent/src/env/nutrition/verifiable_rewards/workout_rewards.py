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

