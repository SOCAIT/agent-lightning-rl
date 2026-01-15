# === Provenance reward (names-only, totals-only) ===
# Matches plan meals to recipe_semantic_search results by NAME only
# and verifies totals are a consistent multiple of tool macros.

import json, math, re, difflib
from statistics import median

# ---------- utils ----------
def _norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # collapse spaces, remove trivial punctuation (& , .  multiple spaces)
    s = s.lower()
    s = re.sub(r"[&]", " and ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)           # keep only letters/digits/spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _rel_err(true, pred):
    if true == 0:
        return 0.0 if abs(pred) < 1e-6 else 1.0
    return abs(pred - true) / max(1.0, abs(true))

def _infer_multiplier(unit: dict, totals: dict, min_dims=2):
    """
    Infer multiplier m from totals using median of ratios across available macros.
    Requires at least `min_dims` macros present. Returns (m, worst_rel_err) or (None, None).
    """
    ratios, dims = [], []
    for k in ("calories", "carbs", "protein", "fat"):
        t = totals.get(k)
        u = unit.get(k)
        if t is None or u in (None, 0):
            continue
        try:
            ratios.append(float(t) / float(u))
            dims.append(k)
        except Exception:
            pass
    if len(ratios) < min_dims:
        return None, None
    m = median(ratios)
    if not (m > 0 and math.isfinite(m)):
        return None, None
    worst = 0.0
    for k in dims:
        pred = unit[k] * m
        err = _rel_err(pred, float(totals[k]))
        if err > worst:
            worst = err
    return m, worst

def _flatten_plan_meals(payload: dict):
    """Expect schema:
    {
      "meals": [
        {"name": "...", "calories": ..., "proteins": ..., "carbs": ..., "fats": ..., "sequence": ...},
        ...
      ]
    }
    """
    meals = []
    if isinstance(payload, dict) and isinstance(payload.get("meals"), list):
        for m in payload["meals"]:
            if isinstance(m, dict):
                meals.append(m)
    return meals

def _extract_totals_from_meal(meal: dict):
    """Map plan keys -> canonical keys (proteins->protein, fats->fat)."""
    totals = {}
    if not isinstance(meal, dict):
        return totals
    if "calories" in meal: totals["calories"] = meal["calories"]
    if "carbs"    in meal: totals["carbs"]    = meal["carbs"]
    if "proteins" in meal: totals["protein"]  = meal["proteins"]
    if "fats"     in meal: totals["fat"]      = meal["fats"]
    return totals

# ---------- catalog from tool logs ----------
def _collect_catalog_from_logs_by_name(traj, tool_name="recipe_semantic_search"):
    """
    Build {normalized_name: {"raw_name": str, "macros": {calories,carbs,protein,fat}, "id": <id>}}
    from tool logs where result looks like:
      [
        {"id": "...", "name": "...", "calories": 382.0, "carbs": 35.0, "protein": 33.0, "fat": 12.0},
        ...
      ]
    or {"recipes": [ ...same list... ]}
    """
    catalog = {}
    for m in getattr(traj, "messages_and_choices", []):
        if m.get("role") != "tool_log":
            continue
        try:
            log = json.loads(m["content"])
        except Exception:
            continue
        end = log.get("end")
        if not end or end.get("tool") != tool_name:
            continue

        res = end.get("result")
        # result might be a JSON string or a Python list/dict
        if isinstance(res, str):
            try:
                res = json.loads(res)
            except Exception:
                pass

        recipes = res.get("recipes") if isinstance(res, dict) else res
        if not isinstance(recipes, list):
            continue

        for r in recipes:
            name = r.get("name")
            if not name:
                continue
            key = _norm_name(name)
            try:
                catalog[key] = {
                    "raw_name": name,
                    "id": r.get("id"),
                    "macros": {
                        "calories": float(r["calories"]),
                        "carbs": float(r["carbs"]),
                        "protein": float(r["protein"]),
                        "fat": float(r["fat"]),
                    },
                }
            except Exception:
                # skip incomplete/bad rows
                continue
    return catalog

# ---------- main reward ----------
def provenance_reward_names_only_totals_only(
    payload: dict,
    traj,
    tol_pct: float = 0.05,
    name_match_cutoff: float = 0.92,
    tool_name: str = "recipe_semantic_search",
):
    """
    Returns (score_in_[0,1], info_dict).

    Pass criteria per meal:
      1) Meal name must match (exact or fuzzy) a recipe name returned by the tool
         during this trajectory.
      2) Meal's totals (calories, carbs, proteins, fats) must be a consistent
         multiple of the tool's macros (worst relative error <= tol_pct).

    No 'servings', no per-serving fields, names-only linkage.
    """
    catalog = _collect_catalog_from_logs_by_name(traj, tool_name=tool_name)
    if not catalog:
        return 0.0, {"reason": "no_recipe_tool_usage_detected"}

    meals = _flatten_plan_meals(payload)
    if not meals:
        return 0.0, {"reason": "empty_plan"}

    checked = passed = 0
    details = []

    # precompute name list for fuzzy matching
    cat_names = list(catalog.keys())

    for meal in meals:
        checked += 1
        plan_name_raw = meal.get("name", "")
        plan_key = _norm_name(plan_name_raw)

        # Try exact normalized match first
        match_key = plan_key if plan_key in catalog else None

        # Fallback: fuzzy match on normalized names
        if match_key is None and cat_names:
            best = difflib.get_close_matches(plan_key, cat_names, n=1, cutoff=name_match_cutoff)
            if best:
                match_key = best[0]

        if match_key is None:
            details.append({
                "ok": False,
                "plan_name": plan_name_raw,
                "reason": "no_name_match_in_tool_results",
            })
            continue

        canon = catalog[match_key]["macros"]
        totals = _extract_totals_from_meal(meal)
        if not totals:
            details.append({
                "ok": False,
                "plan_name": plan_name_raw,
                "matched_recipe_name": catalog[match_key]["raw_name"],
                "reason": "no_totals_in_meal",
            })
            continue

        m, worst_err = _infer_multiplier(canon, totals, min_dims=2)
        ok = (m is not None and worst_err is not None and worst_err <= tol_pct)

        details.append({
            "ok": ok,
            "plan_name": plan_name_raw,
            "matched_recipe_name": catalog[match_key]["raw_name"],
            "matched_recipe_id": catalog[match_key]["id"],
            "inferred_multiplier": m,
            "worst_rel_err": worst_err,
        })

        if ok:
            passed += 1

    score = passed / max(1, checked)
    return score, {"checked": checked, "passed": passed, "items": details}