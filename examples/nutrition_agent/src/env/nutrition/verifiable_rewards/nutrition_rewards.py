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

# 3. Variety Heuristic - Enhanced with protein source and meal type diversity
PROTEIN_KEYWORDS = {
    "chicken": ["chicken", "poultry", "hen"],
    "beef": ["beef", "steak", "burger", "brisket"],
    "fish": ["salmon", "tuna", "fish", "cod", "tilapia", "shrimp", "seafood", "prawn", "trout", "mackerel"],
    "plant": ["chickpea", "lentil", "tofu", "tempeh", "bean", "legume", "vegan", "vegetarian", "quinoa", "edamame"],
    "pork": ["pork", "bacon", "ham", "sausage"],
    "eggs": ["egg", "omelette", "frittata", "scramble"],
    "dairy": ["cheese", "yogurt", "cottage", "paneer"],
    "turkey": ["turkey"],
}

MEAL_TYPE_KEYWORDS = {
    "breakfast": ["oats", "oatmeal", "pancake", "waffle", "eggs", "toast", "smoothie", 
                  "cereal", "muesli", "granola", "yogurt", "breakfast", "morning"],
    "lunch": ["sandwich", "wrap", "salad", "soup", "bowl", "burger", "lunch"],
    "dinner": ["steak", "pasta", "stir-fry", "roast", "curry", "dinner", "rice", "noodle"],
}


def _detect_protein_source(name: str) -> str:
    """Detect the primary protein source from a meal name."""
    name_lower = name.lower()
    for source, keywords in PROTEIN_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return source
    return "unknown"


def _detect_meal_type(name: str) -> str:
    """Detect the meal type from a meal name."""
    name_lower = name.lower()
    for meal_type, keywords in MEAL_TYPE_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return meal_type
    return "other"


def verify_variety_heuristic(payload: dict, strict: bool = True) -> Tuple[float, Dict]:
    """
    GRADUAL variety scoring - gives partial credit so model can learn incrementally.
    
    Evaluates:
    1. Number of meals (partial credit from 1+, ideal 4+)
    2. Unique meal names (partial credit from 1+, ideal 3+)
    3. Protein source diversity (partial credit from 1+, ideal 3+)
    4. Meal type diversity (bonus for spread)
    
    IMPORTANT: This is NOT a hard gate. The model gets partial credit for
    partial variety so it can learn to improve incrementally.
    
    Args:
        payload: The meal plan
        strict: If True, uses stricter scoring curve
        
    Returns:
        score: 0.0 to 1.0 based on variety quality (GRADUAL, not binary)
        info: Details about variety assessment
    """
    meals = payload.get("meals", [])
    if not meals: 
        return 0.0, {"reason": "no_meals"}
    
    num_meals = len(meals)
    
    # Extract meal info
    names = [m.get("name", "").lower().strip() for m in meals]
    unique_names = set(n for n in names if n)  # Filter empty names
    
    # Detect protein sources
    protein_sources = [_detect_protein_source(name) for name in names]
    unique_proteins = set(p for p in protein_sources if p != "unknown")
    
    # Detect meal types
    meal_types = [_detect_meal_type(name) for name in names]
    unique_types = set(t for t in meal_types if t != "other")
    
    info = {
        "total_meals": num_meals,
        "unique_meals": len(unique_names),
        "unique_proteins": len(unique_proteins),
        "protein_sources": list(unique_proteins),
        "unique_meal_types": len(unique_types),
        "meal_types": list(unique_types),
    }
    
    # ========== GRADUAL SCORING ==========
    # Each component contributes to the score proportionally
    # This ensures the model always has a gradient to learn from
    
    score = 0.0
    
    # 1. Number of meals (0.25 weight)
    # 1 meal = 0.05, 2 meals = 0.10, 3 meals = 0.18, 4+ meals = 0.25
    meal_score = min(num_meals / 4.0, 1.0) * 0.25
    score += meal_score
    info["meal_score"] = round(meal_score, 3)
    
    # 2. Unique meal names (0.30 weight) - most important for variety
    # 1 unique = 0.10, 2 unique = 0.20, 3+ unique = 0.30
    name_score = min(len(unique_names) / 3.0, 1.0) * 0.30
    score += name_score
    info["name_score"] = round(name_score, 3)
    
    # 3. Protein source diversity (0.30 weight) - critical for nutrition variety
    # 0 known = 0.05 (baseline), 1 source = 0.15, 2 sources = 0.25, 3+ sources = 0.30
    if len(unique_proteins) == 0:
        protein_score = 0.05  # Small baseline even with unknown proteins
    else:
        protein_score = min(len(unique_proteins) / 3.0, 1.0) * 0.30
    score += protein_score
    info["protein_score"] = round(protein_score, 3)
    
    # 4. Meal type diversity (0.15 weight) - bonus for breakfast/lunch/dinner spread
    # 0 types = 0.0, 1 type = 0.08, 2+ types = 0.15
    type_score = min(len(unique_types) / 2.0, 1.0) * 0.15
    score += type_score
    info["type_score"] = round(type_score, 3)
    
    # ========== STRICT MODE PENALTIES ==========
    if strict:
        # Apply a penalty multiplier if minimums not met (but don't zero out!)
        penalty = 1.0
        
        if num_meals < 3:
            penalty *= 0.7  # 30% penalty for too few meals
            info["penalty_reason"] = info.get("penalty_reason", []) + [f"few_meals_{num_meals}"]
        
        if len(unique_names) < 3:
            penalty *= 0.7  # 30% penalty for low name variety
            info["penalty_reason"] = info.get("penalty_reason", []) + [f"few_unique_names_{len(unique_names)}"]
        
        if len(unique_proteins) < 2:
            penalty *= 0.8  # 20% penalty for low protein variety
            info["penalty_reason"] = info.get("penalty_reason", []) + [f"few_proteins_{len(unique_proteins)}"]
        
        score *= penalty
        info["penalty_multiplier"] = round(penalty, 3)
    
    # Ensure score is in valid range
    final_score = max(0.0, min(1.0, score))
    info["final_variety_score"] = round(final_score, 3)
    
    return final_score, info

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
def combined_reward_v2(
    payload: dict, 
    scenario_data: Scenario, 
    traj=None, 
    skip_llm_judge: bool = False,
    verbose: bool = True,
    strict_variety: bool = True,
):
    """
    Calculate combined reward for a meal plan.
    
    Args:
        payload: The meal plan payload
        scenario_data: Scenario with targets and constraints
        traj: Optional trajectory for provenance tracking
        skip_llm_judge: If True, skip the LLM variety judge (faster for batch processing)
        verbose: If True, print debug info
        strict_variety: If True, requires protein source diversity and higher thresholds
        
    Returns:
        final_score: Float between 0 and 1
        info: Dict with component scores and details
    """
    # Ensure payload is a dict
    payload = get_payload(payload)

    # Force unwrap "answer" if present at top level (this is the key fix)
    if isinstance(payload, dict) and "answer" in payload and isinstance(payload["answer"], dict):
        # Check if inner has "meals"
        if "meals" in payload["answer"]:
            payload = payload["answer"]
        # Or if it's just a wrapper
        elif len(payload) == 1:
             payload = payload["answer"]
    
    # 1. Schema
    r_schema, info_schema = verify_schema_v2(payload)
    if r_schema < 1.0:
        return 0.0, {"failure": "schema", "info": info_schema}
        
    # 2. Macros
    targets = {
        "calories": scenario_data.daily_cal_target,
        "protein": scenario_data.daily_prot_target,
        "carbs": scenario_data.daily_carb_target,
        "fat": scenario_data.daily_fat_target
    }
         
    r_macro, info_macro = verify_macros_strict(payload, targets, tolerance=0.05)
    
    if verbose:
        print(f"DEBUG: Macro check. Targets: {targets}")
        print(f"DEBUG: Macro result: score={r_macro}, info={info_macro}")
    
    # 3. Variety Heuristic (with strict mode for protein diversity)
    r_variety_h, info_variety = verify_variety_heuristic(payload, strict=strict_variety)
    if verbose:
        print(f"DEBUG: Variety heuristic result: score={r_variety_h}, info={info_variety}")
    
    # 4. LLM Judge (Only if basic checks pass and not skipped)
    r_variety_llm = 0.0
    if r_macro > 0.0 and r_variety_h > 0.0 and not skip_llm_judge:
        judge_res = llm_variety_judge(scenario_data.question, payload)
        r_variety_llm = judge_res.get("score", 0.0)
        info_variety["llm_reason"] = judge_res.get("reason")
        if verbose:
            print(f"DEBUG: LLM Judge result: score={r_variety_llm}, reason={info_variety['llm_reason']}")
    elif skip_llm_judge and r_macro > 0.0 and r_variety_h > 0.0:
        # When skipping LLM judge, use the variety heuristic score as proxy
        # The new heuristic already gives a quality-based score (0.4 to 1.0)
        r_variety_llm = r_variety_h  # Use heuristic score as proxy
        info_variety["llm_reason"] = "skipped_using_heuristic_score"

    # ========== MULTIPLICATIVE SCORING ==========
    # BOTH macros AND variety must be good to get a high score.
    # If either is low, the overall score is low.
    #
    # Formula: final = (macro_score * variety_score) ^ 0.5 * scaling
    # This is essentially a geometric mean - both must be high.
    #
    # Examples:
    # - macro=1.0, variety=1.0 → 1.0 (perfect)
    # - macro=1.0, variety=0.5 → ~0.71
    # - macro=0.5, variety=1.0 → ~0.71  
    # - macro=1.0, variety=0.2 → ~0.45
    # - macro=0.5, variety=0.5 → 0.5
    # - macro=0.0, variety=1.0 → 0.0
    
    # Combine variety scores (heuristic + LLM proxy)
    # Weight: 60% heuristic, 40% LLM/proxy
    combined_variety = (0.6 * r_variety_h) + (0.4 * r_variety_llm)
    
    # Geometric mean of macro and variety - BOTH must be high
    # Add small epsilon to avoid zero gradients at boundaries
    epsilon = 0.01
    geometric_mean = ((r_macro + epsilon) * (combined_variety + epsilon)) ** 0.5
    
    # Scale to [0, 1] range (subtract epsilon contribution)
    final_score = max(0.0, geometric_mean - epsilon)
    
    # Minimum score for valid schema (so model always gets some gradient)
    final_score = max(final_score, 0.05)
    
    info = {
        "r_schema": r_schema,
        "r_macro": round(r_macro, 3),
        "r_variety_h": round(r_variety_h, 3),
        "r_variety_llm": round(r_variety_llm, 3),
        "combined_variety": round(combined_variety, 3),
        "geometric_mean": round(geometric_mean, 3),
        "macro_info": info_macro,
        "variety_info": info_variety
    }
    
    return round(final_score, 3), info
