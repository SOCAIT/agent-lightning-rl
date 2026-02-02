import json
import math
from typing import Tuple, Dict, Any, List, Optional, Union
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt
from litellm import completion
from src.nutrition.data_utils import Scenario

VARIETY_JUDGE_PROMPT = """
You are a nutrition expert judging the variety of a meal plan.
Analyze the provided meal plan for:
1. Diversity of ingredients (proteins, vegetables, carb sources).
2. Repetition of meals (is it just the same meal repeated?).
3. Culinary interest.

Score the variety from 0.0 to 1.0.
- 0.0: Extremely repetitive (e.g. same meal 3 times).
- 0.5: Acceptable but basic.
- 1.0: Excellent variety and balance.

Provide a short reason.
"""

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
    
    # Unwrap nested answer keys recursively
    # We loop until no more "answer" wrapper exists or it's not a dict
    while isinstance(obj, dict) and "answer" in obj:
        candidate = obj.get("answer")
        if isinstance(candidate, (dict, list)):
            obj = candidate
        else:
            # If "answer" is a string, try to parse it
            if isinstance(candidate, str):
                try:
                    parsed = json.loads(candidate)
                    obj = parsed
                except:
                    # If parse fails, maybe we just stop unwrapping
                    break
            else:
                break

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

# 1. Schema Check
def verify_schema_v2(payload: dict) -> Tuple[float, Dict]:
    if not isinstance(payload, dict):
        return 0.0, {"error": "payload_not_dict"}
    
    if "meals" not in payload:
        return 0.0, {"error": "missing_meals_key"}
        
    meals = payload["meals"]
    if not isinstance(meals, list):
        return 0.0, {"error": "meals_not_list"}
        
    if len(meals) == 0:
        return 0.0, {"error": "empty_meals_list"}
        
    required_keys = ["name", "quantity", "calories", "proteins", "carbs", "fats"]
    for i, meal in enumerate(meals):
        if not isinstance(meal, dict):
            return 0.0, {"error": f"meal_{i}_not_dict"}
        for k in required_keys:
            if k not in meal:
                return 0.0, {"error": f"meal_{i}_missing_{k}"}
                
    return 1.0, {"status": "valid"}

# 2. GRADUAL Macro Check - gives partial credit for being close!
def verify_macros_strict(payload: dict, targets: Dict[str, float], tolerance=0.05) -> Tuple[float, Dict]:
    """
    GRADUAL macro scoring - gives partial credit so model can learn incrementally.
    
    Instead of binary pass/fail, we score based on how close each macro is to target.
    This ensures the model always has a gradient to learn from.
    
    Scoring per macro:
    - Within tolerance (e.g., 5%): 1.0
    - 2x tolerance (e.g., 10%): 0.7
    - 3x tolerance (e.g., 15%): 0.4
    - 4x+ tolerance (e.g., 20%+): 0.1 minimum (always some gradient)
    
    Final score is weighted average of individual macro scores.
    """
    meals = payload.get("meals", [])
    if not meals: return 0.0, {"error": "no_meals"}
    
    total_cals = sum(float(m.get("calories", 0)) for m in meals)
    total_prot = sum(float(m.get("proteins", 0)) for m in meals)
    total_carb = sum(float(m.get("carbs", 0)) for m in meals)
    total_fat = sum(float(m.get("fats", 0)) for m in meals)
    
    errors = {}
    macro_scores = {}
    weights = {}
    
    def _score_macro(actual: float, target: float, tol: float, name: str, weight: float) -> float:
        """Score a single macro with gradual falloff."""
        if target <= 0:
            return 1.0  # No target = automatic pass
        
        err = abs(actual - target) / target
        errors[name] = round(err, 4)
        
        # Gradual scoring based on error relative to tolerance
        if err <= tol:
            # Within tolerance: full score
            score = 1.0
        elif err <= tol * 2:
            # 1-2x tolerance: linear falloff from 1.0 to 0.7
            score = 1.0 - 0.3 * ((err - tol) / tol)
        elif err <= tol * 3:
            # 2-3x tolerance: linear falloff from 0.7 to 0.4
            score = 0.7 - 0.3 * ((err - tol * 2) / tol)
        elif err <= tol * 4:
            # 3-4x tolerance: linear falloff from 0.4 to 0.1
            score = 0.4 - 0.3 * ((err - tol * 3) / tol)
        else:
            # Beyond 4x tolerance: minimum score (preserves gradient)
            score = 0.1
        
        macro_scores[name] = round(score, 3)
        weights[name] = weight
        return score
    
    # Score each macro with appropriate weights
    # Calories and protein are most important (higher weights)
    cal_score = _score_macro(total_cals, float(targets.get("calories", 0)), tolerance, "calories", 0.4)
    prot_score = _score_macro(total_prot, float(targets.get("protein", 0)), tolerance, "protein", 0.4)
    
    # Carbs and fat are optional/less critical
    carb_score = 1.0
    fat_score = 1.0
    if targets.get("carbs"):
        carb_score = _score_macro(total_carb, float(targets["carbs"]), tolerance * 1.5, "carbs", 0.1)
    if targets.get("fat"):
        fat_score = _score_macro(total_fat, float(targets["fat"]), tolerance * 1.5, "fat", 0.1)
    
    # Calculate weighted average
    total_weight = sum(weights.values()) if weights else 1.0
    if total_weight > 0:
        weighted_sum = sum(macro_scores.get(k, 1.0) * w for k, w in weights.items())
        final_score = weighted_sum / total_weight
    else:
        final_score = (cal_score + prot_score) / 2.0
    
    # Determine if "passed" (for backwards compatibility) - all within tolerance
    passed = all(errors.get(k, 0) <= tolerance for k in ["calories", "protein"] if k in errors)
    
    return round(final_score, 3), {
        "totals": {"cal": round(total_cals, 1), "prot": round(total_prot, 1), "carb": round(total_carb, 1), "fat": round(total_fat, 1)},
        "errors": errors,
        "macro_scores": macro_scores,
        "passed": passed,
        "gradual_score": round(final_score, 3)
    }

# 3. GRADUAL Variety Heuristic - gives partial credit
def verify_variety_heuristic(payload: dict) -> Tuple[float, Dict]:
    """
    GRADUAL variety scoring - gives partial credit so model can learn incrementally.
    
    Scoring:
    - 1 meal: 0.2
    - 2 meals: 0.4
    - 3+ meals: 0.6 base
    - Unique names add up to +0.4 more
    """
    meals = payload.get("meals", [])
    if not meals: 
        return 0.0, {"reason": "no_meals"}
    
    num_meals = len(meals)
    names = [m.get("name", "").lower().strip() for m in meals]
    unique_names = set(n for n in names if n)
    
    info = {
        "total_meals": num_meals,
        "unique_meals": len(unique_names),
    }
    
    # Gradual scoring for number of meals (up to 0.5)
    meal_score = min(num_meals / 4.0, 1.0) * 0.5
    
    # Gradual scoring for unique names (up to 0.5)
    name_score = min(len(unique_names) / 3.0, 1.0) * 0.5
    
    final_score = meal_score + name_score
    
    info["meal_score"] = round(meal_score, 3)
    info["name_score"] = round(name_score, 3)
    info["final_variety_score"] = round(final_score, 3)
    
    return round(final_score, 3), info

# 4. LLM Variety Judge
class VarietyJudgeResponse(BaseModel):
    score: float
    reason: str

@retry(stop=stop_after_attempt(2))
def llm_variety_judge(scenario_text, plan_json):
    messages = [
        {"role": "system", "content": VARIETY_JUDGE_PROMPT},
        {"role": "user", "content": f"Context: {scenario_text}\n\nPlan: {json.dumps(plan_json)}"}
    ]
    try:
        response = completion(
            model="openai/gpt-4o-mini", # Or use the local model if needed
            messages=messages,
            response_format=VarietyJudgeResponse
        )
        content = response.choices[0].message.content
        # If response_format handled it, we get a json string
        return json.loads(content)
    except Exception as e:
        print(f"Judge Error: {e}")
        return {"score": 0.5, "reason": "judge_error"}

# Main Reward Wrapper
def combined_reward_v2(payload: dict, scenario_data: Scenario, traj=None, skip_llm_judge: bool = True):
    """
    Calculate combined reward with GRADUAL macro scoring.
    
    Now that macros use gradual scoring, the reward function provides
    continuous feedback for the model to learn from.
    """
    # Ensure payload is a dict
    payload = get_payload(payload)
    
    # 1. Schema
    r_schema, info_schema = verify_schema_v2(payload)
    if r_schema < 1.0:
        return 0.0, {"failure": "schema", "info": info_schema}
        
    # 2. Macros (NOW GRADUAL!)
    targets = {
        "calories": scenario_data.daily_cal_target,
        "protein": scenario_data.daily_prot_target,
        "carbs": scenario_data.daily_carb_target,
        "fat": scenario_data.daily_fat_target
    }
    r_macro, info_macro = verify_macros_strict(payload, targets, tolerance=0.05)
    
    # 3. Variety Heuristic
    r_variety_h, info_variety = verify_variety_heuristic(payload)
    
    # 4. LLM Judge (skip by default for training speed)
    r_variety_llm = 0.0
    if not skip_llm_judge and r_macro > 0.3 and r_variety_h > 0.3:
         judge_res = llm_variety_judge(scenario_data.question, payload)
         r_variety_llm = judge_res.get("score", 0.0)
         info_variety["llm_reason"] = judge_res.get("reason")
    elif skip_llm_judge:
        # Use variety heuristic as proxy for LLM judge
        r_variety_llm = r_variety_h
        info_variety["llm_reason"] = "skipped_using_heuristic"

    # ========== GRADUAL WEIGHTED SCORING ==========
    # Now that r_macro is gradual (0.1 to 1.0), we can do proper weighted sum
    # 
    # Weights:
    # - Macro accuracy: 50% (most important for nutrition)
    # - Variety heuristic: 30%
    # - LLM variety proxy: 20%
    
    final_score = (0.5 * r_macro) + (0.3 * r_variety_h) + (0.2 * r_variety_llm)
    
    # Ensure minimum score for valid schema (preserves gradient)
    final_score = max(final_score, 0.05)
        
    info = {
        "r_schema": r_schema,
        "r_macro": round(r_macro, 3),
        "r_variety_h": round(r_variety_h, 3),
        "r_variety_llm": round(r_variety_llm, 3),
        "macro_info": info_macro,
        "variety_info": info_variety
    }
    
    return round(final_score, 3), info
