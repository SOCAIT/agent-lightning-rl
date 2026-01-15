from src.env.nutrition.verifiable_rewards import verify_schema_v2, verify_macros_strict, verify_variety_heuristic

# Main Reward Wrapper
async def combined_reward_v2(payload: dict, scenario_data: Scenario, traj):
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
    
    # 3. Variety Heuristic
    r_variety_h, info_variety = verify_variety_heuristic(payload)
    
    # 4. LLM Judge (Only if basic checks pass to save cost/time)
    r_variety_llm = 0.0
    if r_macro > 0.0 and r_variety_h > 0.0:
         judge_res = await llm_variety_judge(scenario_data.question, payload)
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