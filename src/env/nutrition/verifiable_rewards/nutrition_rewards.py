import math
import re
from src.env.verifiable_rewards.schema_rewards import verify_meal_plan_schema, verify_workout_schema, verify_nutrition_schema
from src.env.verifiable_rewards.banned_food_reward import verify_no_banned

import math

# ---------- smooth per-macro scoring ----------
def _smooth_macro_score(rel_err: float, cap: float = 0.15) -> float:
    """
    Dense, smooth score in (0,1]. rel_err is |y - t| / max(1, |t|).
    cap is the 'scale' where score ~ 1/(1+3) ~= 0.25 if rel_err == cap.
    Smaller cap => stricter. Use curriculum to anneal cap down over training.
    """
    x = max(0.0, rel_err) / max(1e-9, cap)
    return 1.0 / (1.0 + 3.0 * (x * x))

def _macro_caps_and_weights(step: int | None):
    """
    Anneal strictness and emphasize protein slightly.
    Returns (caps, weights) as dicts keyed by 'calories','proteins','carbs','fats'.
    """
    if step is None or step < 100:
        caps = {"calories": 0.20, "proteins": 0.18, "carbs": 0.25, "fats": 0.25}
        w    = {"calories": 0.30, "proteins": 0.40, "carbs": 0.20, "fats": 0.10}
    elif step < 300:
        caps = {"calories": 0.15, "proteins": 0.12, "carbs": 0.20, "fats": 0.20}
        w    = {"calories": 0.30, "proteins": 0.40, "carbs": 0.20, "fats": 0.10}
    else:
        caps = {"calories": 0.10, "proteins": 0.08, "carbs": 0.15, "fats": 0.15}
        w    = {"calories": 0.30, "proteins": 0.40, "carbs": 0.20, "fats": 0.10}
    # normalize weights
    s = sum(w.values())
    for k in w: w[k] = w[k] / s
    return caps, w

def verify_daily_meal_plan_macros_v2(
    plan_json: dict,
    daily_cal_target: float,
    daily_prot_target: float,
    daily_carb_target: float | None = None,
    daily_fat_target: float | None = None,
    step: int | None = None,
):
    """
    Dense macro scorer using all available targets (cal/proteins and optionally carbs/fats).
    Returns (score in [0,1], diag).
    """
    meals = plan_json.get("meals", [])
    totals = {
        "calories": sum(float(m.get("calories", 0)  or 0) for m in meals),
        "proteins": sum(float(m.get("proteins", 0)  or 0) for m in meals),
        "carbs":    sum(float(m.get("carbs", 0)     or 0) for m in meals),
        "fats":     sum(float(m.get("fats", 0)      or 0) for m in meals),
    }
    targets = {
        "calories": float(daily_cal_target),
        "proteins": float(daily_prot_target),
        "carbs":    float(daily_carb_target) if daily_carb_target is not None else None,
        "fats":     float(daily_fat_target)  if daily_fat_target  is not None else None,
    }

    caps, weights = _macro_caps_and_weights(step)

    per_macro = {}
    used_keys = []
    for k in ("calories", "proteins", "carbs", "fats"):
        if targets[k] is None:
            continue
        rel = abs(totals[k] - targets[k]) / max(1.0, abs(targets[k]))
        s   = _smooth_macro_score(rel, cap=caps[k])
        per_macro[k] = {"rel_err": rel, "cap": caps[k], "score": s, "weight": weights[k]}
        used_keys.append(k)

    if not used_keys:  # safeguard
        return 0.0, {"error": "no macro targets provided"}

    score = sum(per_macro[k]["score"] * per_macro[k]["weight"] for k in used_keys)

    diag = {
        "totals": totals,
        "targets": {k: targets[k] for k in used_keys},
        "per_macro": {k: per_macro[k] for k in used_keys},
        "used_keys": used_keys,
    }
    return max(0.0, min(1.0, score)), diag

def _banded_score(err, tol=0.03, hard=0.10):
    """
    err: relative error (abs(actual-target)/target).
    1.0 inside tol; cosine-decay to 0 by 'hard'.
    """
    if err <= tol: return 1.0
    if err >= hard: return 0.0
    x = (err - tol) / (hard - tol)
    return 0.5 * (1 + math.cos(math.pi * x))

def verify_daily_meal_plan_macros(plan_json, daily_cal_target, daily_prot_target,
                                  tol_cal=0.03, hard_cal=0.10,
                                  tol_pro=0.02, hard_pro=0.08):
    """
    Returns (score in [0,1], diagnostics).
    Score is a weighted combo emphasizing protein.
    """
    meals = plan_json.get("meals", [])
    tot_c = sum(m.get("calories", 0) for m in meals)
    tot_p = sum(m.get("proteins", 0) for m in meals)

    rel_c = abs(tot_c - daily_cal_target) / max(daily_cal_target, 1)
    rel_p = abs(tot_p - daily_prot_target) / max(daily_prot_target, 1)

    s_cal = _banded_score(rel_c, tol=tol_cal, hard=hard_cal)
    s_pro = _banded_score(rel_p, tol=tol_pro, hard=hard_pro)

    # Emphasize protein, then calories
    score = 0.6 * s_pro + 0.4 * s_cal

    diag = {
        "totals": {"calories": tot_c, "proteins": tot_p},
        "targets": {"calories": daily_cal_target, "proteins": daily_prot_target},
        "rel_errors": {"cal": rel_c, "pro": rel_p},
        "component_scores": {"cal": s_cal, "pro": s_pro},
        "within_5pct": {"cal": rel_c <= 0.05, "pro": rel_p <= 0.05},
    }
    return score, diag

def verify_macros(plan_json, daily_cal_target, daily_prot_target,
                  tol_cal=0.03, hard_cal=0.10, tol_pro=0.02, hard_pro=0.08): #, tol_carbs=0.05, hard_carbs=0.10, tol_fats=0.05, hard_fats=0.10):
    """
    plan_json["dailyMealPlans"] is a list of days.
    Returns (avg_score, per_day_diags).
    """
    days = plan_json.get("dailyMealPlans", [])
    if not isinstance(days, list) or not days:
        return 0.0, {"error": "dailyMealPlans missing/empty"}

    per_day = []
    for day in days:
        meals = day.get("meals", [])
        tot_c = sum(m.get("calories", 0) for m in meals)
        tot_p = sum(m.get("proteins", 0) for m in meals)
        rel_c = abs(tot_c - daily_cal_target) / max(daily_cal_target, 1)
        rel_p = abs(tot_p - daily_prot_target) / max(daily_prot_target, 1)
        s_cal = _banded_score(rel_c, tol=tol_cal, hard=hard_cal)
        s_pro = _banded_score(rel_p, tol=tol_pro, hard=hard_pro)
        score = 0.6*s_pro + 0.4*s_cal
        per_day.append({"day": day.get("day"), "score": score, "tot_c": tot_c, "tot_p": tot_p, "rel_c": rel_c, "rel_p": rel_p})
    avg = sum(d["score"] for d in per_day) / len(per_day)
    return avg, {"per_day": per_day}


# def verify_no_banned(plan_json, banned_keywords):
#     """
#     Ensure no banned keywords appear in meal name or description.
#     Returns list of (day, meal_name, offending_keyword) if violations.
#     """
#     violations = []
#     R_banned = 1.0
#     for day in plan_json.get("dailyMealPlans", []):
#         d = day.get("day")
#         for m in day.get("meals", []):
#             text = (m.get("name", "") + " " + m.get("description", "")).lower()
#             for kw in banned_keywords:
#                 if re.search(rf"\b{re.escape(kw.lower())}\b", text):
#                     violations.append((d, m.get("name", ""), kw))
#                     R_banned = 0.0
#     return R_banned, violations


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



def nutrition_reward(
    payload,
    daily_cal_target,
    daily_prot_target,
    banned_keywords,
    traj=None,
    daily_carb_target=None,
    daily_fat_target=None,
    step: int | None = None,
):
    # schema check
    R_schema, diag_schema = verify_meal_plan_schema(payload)

    # DENSE macros (uses carbs/fats if provided; anneals by step)
    R_macro,  diag_macro  = verify_daily_meal_plan_macros_v2(
        payload,
        daily_cal_target=daily_cal_target,
        daily_prot_target=daily_prot_target,
        daily_carb_target=daily_carb_target,
        daily_fat_target=daily_fat_target,
        step=step,
    )

    # banned check (same as yours)
    if banned_keywords:
        R_banned, diag_banned = verify_no_banned(payload, banned_keywords)
    else:
        R_banned, diag_banned = 1.0, {"ok": True}

    # weights: lean harder on macros; keep schema meaningful
    if banned_keywords:
        reward_weights = {"R_macro": 0.75, "R_schema": 0.20, "R_banned": 0.05}
    else:
        reward_weights = {"R_macro": 0.80, "R_schema": 0.20, "R_banned": 0.00}

    base = (
        reward_weights["R_macro"]  * R_macro  +
        reward_weights["R_schema"] * R_schema +
        reward_weights["R_banned"] * R_banned
    )

    total = max(0.0, min(1.0, base))  # keep it cleanly in [0,1]
    diag = {
        "R_macro": R_macro, "R_schema": R_schema, "R_banned": R_banned,
        "diag_macro": diag_macro, "diag_schema": diag_schema, "diag_banned": diag_banned,
        "weights": reward_weights,
    }
    return total, diag

def nutrition_reward_weekly(payload, daily_cal_target, daily_prot_target, banned_keywords, traj=None):
    R_schema , diag_schema = verify_nutrition_schema(payload)
    R_macro,  diag_macro  = verify_macros(payload, daily_cal_target, daily_prot_target)
    if banned_keywords:
        R_banned, diag_banned = verify_no_banned(payload, banned_keywords)
    else:
        R_banned = 1.0
        diag_banned = {"ok": True}

    # Heavier weight on macros
    if banned_keywords:
        reward_weights = {  "R_macro": 0.70, "R_schema": 0.20,  "R_banned": 0.10}
    else:
        reward_weights = { "R_macro": 0.70, "R_schema": 0.30, "R_banned": 0.00  }

    F_final = 1.0
   
    base = reward_weights["R_macro"]*R_macro + reward_weights["R_schema"]*R_schema + reward_weights["R_banned"]*R_banned
    total = max(0.0, min(1.05, base * F_final))
    diag = {"R_macro": R_macro, "R_schema": R_schema, "R_banned": R_banned, "F_final": F_final,
            "diag_macro": diag_macro, "diag_schema": diag_schema, "diag_banned": diag_banned}
    return total, diag

