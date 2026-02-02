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
    print(f"DEBUG: verify_schema_v2 received payload keys: {list(payload.keys())}")
    if not isinstance(payload, dict):
        return 0.0, {"error": "payload_not_dict"}
    
    if "meals" not in payload:
        print(f"DEBUG: verify_schema_v2 missing 'meals'. Payload: {str(payload)[:500]}")
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

# 2. Strict Macro Check (+/- 5%)
def verify_macros_strict(payload: dict, targets: Dict[str, float], tolerance=0.05) -> Tuple[float, Dict]:
    meals = payload.get("meals", [])
    if not meals: return 0.0, {"error": "no_meals"}
    
    total_cals = sum(float(m.get("calories", 0)) for m in meals)
    total_prot = sum(float(m.get("proteins", 0)) for m in meals)
    total_carb = sum(float(m.get("carbs", 0)) for m in meals)
    total_fat = sum(float(m.get("fats", 0)) for m in meals)
    
    errors = {}
    passed = True
    
    # Check Calories
    if targets.get("calories"):
        target = float(targets["calories"])
        if target > 0:
            err = abs(total_cals - target) / target
            errors["calories"] = err
            if err > tolerance: passed = False
            
    # Check Protein
    if targets.get("protein"):
        target = float(targets["protein"])
        if target > 0:
            err = abs(total_prot - target) / target
            errors["protein"] = err
            if err > tolerance: passed = False
            
    # Optional checks if targets exist
    if targets.get("carbs"):
        target = float(targets["carbs"])
        if target > 0:
            err = abs(total_carb - target) / target
            errors["carbs"] = err
            if err > tolerance and tolerance > 0.0: passed = False
            
    if targets.get("fat"):
        target = float(targets["fat"])
        if target > 0:
            err = abs(total_fat - target) / target
            errors["fat"] = err
            if err > tolerance and tolerance > 0.0: passed = False
            
    score = 1.0 if passed else 0.0
    return score, {"totals": {"cal": total_cals, "prot": total_prot}, "errors": errors, "passed": passed}

# 3. Variety Heuristic
def verify_variety_heuristic(payload: dict) -> Tuple[float, Dict]:
    meals = payload.get("meals", [])
    if not meals: return 0.0, {"reason": "no_meals"}
    
    # Check number of meals (aim for 3+)
    num_meals = len(meals)
    if num_meals < 3:
        return 0.0, {"reason": f"too_few_meals_{num_meals}"}
        
    # Check unique meal names (aim for 3+)
    names = [m.get("name", "").lower().strip() for m in meals]
    unique_names = set(names)
    if len(unique_names) < 3:
        return 0.0, {"reason": f"low_variety_{len(unique_names)}_unique"}
        
    return 1.0, {"unique_meals": len(unique_names), "total_meals": num_meals}

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
def combined_reward_v2(payload: dict, scenario_data: Scenario, traj=None):
    # Ensure payload is a dict
    payload = get_payload(payload)

    print(f"got payload: {payload}")
    
    # 1. Schema
    # r_schema, info_schema = verify_schema_v2(payload)
    # if r_schema < 1.0:
    #     return 0.0, {"failure": "schema", "info": info_schema}
    r_schema = 1.0
    info_schema = {"status": "skipped"}
        
    # 2. Macros
    targets = {
        "calories": scenario_data.daily_cal_target,
        "protein": scenario_data.daily_prot_target,
        "carbs": scenario_data.daily_carb_target,
        "fat": scenario_data.daily_fat_target
    }
    r_macro, info_macro = verify_macros_strict(payload, targets, tolerance=0.05)
    
    # 3. Variety Heuristic
    r_variety_h, info_variety = verify_variety_heuristic(payload)
    
    # 4. LLM Judge (Only if basic checks pass to save cost/time)
    r_variety_llm = 0.0
    if r_macro > 0.0 and r_variety_h > 0.0:
         judge_res = llm_variety_judge(scenario_data.question, payload)
         r_variety_llm = judge_res.get("score", 0.0)
         info_variety["llm_reason"] = judge_res.get("reason")
    else:
        # Penalize if basic variety check fails
        pass

    # Weighted Sum
    # Macro accuracy is paramount -> 0.4
    # Schema is a gate (already handled, if 0 return 0)
    # Variety Heuristic -> 0.3
    # LLM Variety -> 0.3
    
    # If macros fail, score is low
    if r_macro == 0.0:
        final_score = 0.1 # participation award
    else:
        # Base score from macros
        final_score = 0.4 
        # Add variety
        final_score += (0.3 * r_variety_h)
        # Add LLM score
        final_score += (0.3 * r_variety_llm)
        
    info = {
        "r_schema": r_schema,
        "r_macro": r_macro,
        "r_variety_h": r_variety_h,
        "r_variety_llm": r_variety_llm,
        "macro_info": info_macro,
        "variety_info": info_variety
    }
    
    return final_score, info
