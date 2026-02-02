"""
High-Quality SFT Data Generator for Nutrition Agent

This script generates finetuning data that teaches models to output meal plans
that achieve high rewards (close to 1.0). The key insight is using mathematical
optimization to find optimal meal quantities that hit macro targets precisely.

Key improvements over basic generation:
1. Uses 4 meals per day (not just 3)
2. Individual quantity optimization per meal using scipy.optimize
3. Diversity-aware meal selection (different protein sources, categories)
4. Validates all examples with the reward function, only keeps high-quality ones
5. Generates reasoning traces showing the optimization process

Usage:
    python generate_sft_data.py --input ../../data/fitness_scenarios_train.parquet \
                                --output ../../data/nutrition_sft_data_optimized.jsonl \
                                --min-reward 0.85 \
                                --verbose
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from itertools import combinations

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.optimize import minimize

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project setup - add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import reward function for validation
from src.env.nutrition.verifiable_rewards.nutrition_rewards import (
    combined_reward_v2,
    verify_schema_v2,
    verify_macros_strict,
    verify_variety_heuristic,
)
from src.nutrition.data_utils import Scenario

# ============================================================================
# PINECONE SETUP - Standalone to avoid decorator issues
# ============================================================================

try:
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    recipe_index = pc.Index("syntrafit-meals-nutrition")
    PINECONE_AVAILABLE = True
except Exception as e:
    logger.warning(f"Pinecone not available: {e}")
    PINECONE_AVAILABLE = False
    recipe_index = None


def search_recipes_direct(meal_query: str, k: int = 5) -> List[Dict]:
    """
    Direct Pinecone search without decorators/side effects.
    This is used for SFT data generation to avoid the log_tool decorator issues.
    """
    if not PINECONE_AVAILABLE or recipe_index is None:
        logger.error("Pinecone not available for search")
        return []
    
    try:
        results = recipe_index.search(
            namespace="syntrafit",
            query={
                "top_k": k,
                "inputs": {"text": meal_query}
            }
        )
        
        # Extract meal info from results
        data = results.get("result", {})
        meals = []
        for hit in data.get("hits", []):
            fields = hit.get("fields", {})
            if "name" in fields:
                meals.append({
                    "id": hit.get("_id", ""),
                    "name": str(fields.get("name", ""))[:80],
                    "calories": float(fields.get("calories", 0)),
                    "carbs": float(fields.get("carbs", 0)),
                    "protein": float(fields.get("proteins", 0)),
                    "fat": float(fields.get("fats", 0)),
                })
        return meals
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []


# Prompt and helper functions
PLANNER_PROMPT = """
You are a nutrition planner. Create a ONE-DAY meal plan that matches the user's macros and dietary restrictions.

Tools:
- recipe_semantic_search(meal_query, k)
- return_final_answer_tool(answer)

Rules:
- Always return a single-day plan even if user asks for 7 days.
- Meal names MUST come from recipe_semantic_search results.
- Macros must be based on tool results and scaled by quantity (serving multiplier).
- Final response MUST be a single call to return_final_answer_tool with the JSON plan.
- Tool call arguments MUST be valid JSON. The tool expects an object, not a JSON string.
- Call format: return_final_answer_tool({"answer": {"meals": [ ... ]}})
- JSON schema: {"meals":[{"name":str,"quantity":number,"calories":number,"proteins":number,"carbs":number,"fats":number,"sequence":int}]}
- Do NOT nest meals under "items" or change key names (e.g. "protein" -> "proteins").
"""

def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 12] + " ...(truncated)"


# Constants
MAX_CONTEXT_CHARS = 400
MAX_INPUT_CHARS = 800
MIN_QUANTITY = 0.5
MAX_QUANTITY = 3.0
NUM_MEALS_TARGET = 4
MIN_REWARD_THRESHOLD = 0.85  # Only keep examples with reward >= this


# ============================================================================
# MEAL CATEGORIZATION FOR DIVERSITY
# ============================================================================

BREAKFAST_KEYWORDS = ["oats", "oatmeal", "pancake", "waffle", "eggs", "toast", "smoothie", 
                       "cereal", "muesli", "granola", "yogurt", "breakfast"]
LUNCH_KEYWORDS = ["sandwich", "wrap", "salad", "soup", "bowl", "burger"]
DINNER_KEYWORDS = ["steak", "pasta", "stir-fry", "roast", "curry", "dinner", "rice", "noodle"]
PROTEIN_SOURCES = {
    "chicken": ["chicken", "poultry"],
    "beef": ["beef", "steak", "burger"],
    "fish": ["salmon", "tuna", "fish", "cod", "tilapia", "shrimp", "seafood"],
    "plant": ["chickpea", "lentil", "tofu", "tempeh", "bean", "legume", "vegan", "vegetarian"],
    "pork": ["pork", "bacon", "ham"],
    "eggs": ["egg", "omelette", "frittata"],
    "dairy": ["cheese", "yogurt", "milk"],
}


def categorize_meal(name: str) -> Dict[str, Any]:
    """Categorize a meal by type and protein source."""
    name_lower = name.lower()
    
    # Determine meal type
    meal_type = "other"
    if any(kw in name_lower for kw in BREAKFAST_KEYWORDS):
        meal_type = "breakfast"
    elif any(kw in name_lower for kw in LUNCH_KEYWORDS):
        meal_type = "lunch"
    elif any(kw in name_lower for kw in DINNER_KEYWORDS):
        meal_type = "dinner"
    
    # Determine protein source
    protein_source = "unknown"
    for source, keywords in PROTEIN_SOURCES.items():
        if any(kw in name_lower for kw in keywords):
            protein_source = source
            break
    
    return {"meal_type": meal_type, "protein_source": protein_source}


def select_diverse_meals(
    candidates: List[Dict], 
    num_meals: int = 4,
    min_protein_sources: int = 2,
    banned_keywords: List[str] = None,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Select diverse meals from candidates ensuring variety in:
    1. Protein sources (minimum 2 different sources required)
    2. Meal types
    3. Unique meal names
    
    Args:
        candidates: List of meal candidates
        num_meals: Target number of meals to select
        min_protein_sources: Minimum different protein sources required
        banned_keywords: Keywords to filter out
        
    Returns:
        selected: List of selected meals
        diversity_info: Information about diversity achieved
    """
    banned = {str(k).lower() for k in (banned_keywords or [])}
    
    # Filter and categorize candidates
    categorized = []
    for meal in candidates:
        name = meal.get("name", "")
        # Skip banned foods
        if any(bad in name.lower() for bad in banned):
            continue
        # Skip zero-calorie meals
        if float(meal.get("calories", 0)) <= 0:
            continue
        
        cat = categorize_meal(name)
        categorized.append({
            **meal,
            "_protein": cat["protein_source"],
            "_type": cat["meal_type"],
        })
    
    if len(categorized) < 3:
        # Not enough candidates after filtering
        return [], {"error": "too_few_candidates_after_filtering"}
    
    # Group by protein source
    by_protein: Dict[str, List[Dict]] = {}
    for meal in categorized:
        protein = meal["_protein"]
        if protein not in by_protein:
            by_protein[protein] = []
        by_protein[protein].append(meal)
    
    # Count unique protein sources available (excluding "unknown")
    known_proteins = [p for p in by_protein.keys() if p != "unknown"]
    
    if len(known_proteins) < min_protein_sources:
        # Not enough protein variety available
        return [], {"error": f"only_{len(known_proteins)}_protein_sources_available", "sources": known_proteins}
    
    # Strategy: Greedy selection ensuring protein diversity
    selected = []
    used_proteins = set()
    used_names = set()
    
    # First pass: Get one meal from each protein source (prioritize known proteins)
    for protein in known_proteins:
        if len(selected) >= num_meals:
            break
        for meal in by_protein[protein]:
            name = meal.get("name", "").lower()
            if name not in used_names:
                selected.append(meal)
                used_proteins.add(protein)
                used_names.add(name)
                break
    
    # Second pass: Fill remaining slots, still preferring new proteins
    for protein in list(by_protein.keys()):
        if len(selected) >= num_meals:
            break
        for meal in by_protein[protein]:
            name = meal.get("name", "").lower()
            if name not in used_names:
                selected.append(meal)
                used_proteins.add(protein)
                used_names.add(name)
                if len(selected) >= num_meals:
                    break
    
    # Third pass: If still need more, add any remaining unique meals
    for meal in categorized:
        if len(selected) >= num_meals:
            break
        name = meal.get("name", "").lower()
        if name not in used_names:
            selected.append(meal)
            used_names.add(name)
    
    # Clean up temporary category data
    for meal in selected:
        meal.pop("_protein", None)
        meal.pop("_type", None)
    
    # Verify we have enough protein diversity
    final_proteins = set()
    for meal in selected:
        cat = categorize_meal(meal.get("name", ""))
        if cat["protein_source"] != "unknown":
            final_proteins.add(cat["protein_source"])
    
    diversity_info = {
        "unique_proteins": len(final_proteins),
        "protein_sources": list(final_proteins),
        "unique_meals": len(selected),
        "candidates_available": len(categorized),
    }
    
    if len(final_proteins) < min_protein_sources:
        return [], {**diversity_info, "error": f"could_not_achieve_{min_protein_sources}_protein_sources"}
    
    return selected, diversity_info


# ============================================================================
# QUANTITY OPTIMIZATION
# ============================================================================

def optimize_quantities(
    meals: List[Dict],
    target_calories: float,
    target_protein: float,
    target_carbs: Optional[float] = None,
    target_fat: Optional[float] = None,
    tolerance: float = 0.05,
) -> Tuple[List[float], Dict[str, float]]:
    """
    Use scipy.optimize to find optimal meal quantities that hit macro targets.
    
    This is the key innovation that allows us to generate high-quality training data.
    Instead of uniform scaling, we find individual quantities for each meal.
    
    Returns:
        quantities: List of optimal quantities for each meal
        errors: Dict of relative errors for each macro
    """
    n = len(meals)
    if n == 0:
        return [], {"error": "no_meals"}
    
    # Extract base macros (per 1 serving)
    base_calories = np.array([float(m.get("calories", 0)) for m in meals])
    base_protein = np.array([float(m.get("protein", 0)) for m in meals])
    base_carbs = np.array([float(m.get("carbs", 0)) for m in meals])
    base_fat = np.array([float(m.get("fat", 0)) for m in meals])
    
    # Targets
    targets = np.array([target_calories, target_protein])
    macro_matrix = np.array([base_calories, base_protein])
    
    # Add carbs and fat if targets exist
    if target_carbs and target_carbs > 0:
        targets = np.append(targets, target_carbs)
        macro_matrix = np.vstack([macro_matrix, base_carbs])
    if target_fat and target_fat > 0:
        targets = np.append(targets, target_fat)
        macro_matrix = np.vstack([macro_matrix, base_fat])
    
    def objective(q):
        """Minimize weighted sum of squared relative errors."""
        totals = macro_matrix @ q
        # Avoid division by zero
        safe_targets = np.where(targets > 0, targets, 1)
        relative_errors = (totals - targets) / safe_targets
        # Weight calories and protein more heavily
        weights = np.array([2.0, 2.0] + [1.0] * (len(targets) - 2))
        return np.sum(weights * relative_errors**2)
    
    # Initial guess: equal quantities that roughly hit calorie target
    initial_total_cal = np.sum(base_calories)
    if initial_total_cal > 0:
        initial_scale = target_calories / initial_total_cal
        q0 = np.full(n, np.clip(initial_scale, MIN_QUANTITY, MAX_QUANTITY))
    else:
        q0 = np.ones(n)
    
    # Bounds: each quantity between MIN and MAX
    bounds = [(MIN_QUANTITY, MAX_QUANTITY)] * n
    
    # Optimize
    result = minimize(
        objective,
        q0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-8}
    )
    
    optimal_q = result.x
    
    # Round to 1 decimal place for cleaner output
    optimal_q = np.round(optimal_q, 1)
    
    # Calculate final errors
    final_totals = macro_matrix @ optimal_q
    errors = {}
    error_names = ["calories", "protein"] + (["carbs"] if target_carbs else []) + (["fat"] if target_fat else [])
    for i, name in enumerate(error_names):
        if targets[i] > 0:
            errors[name] = abs(final_totals[i] - targets[i]) / targets[i]
        else:
            errors[name] = 0.0
    
    return optimal_q.tolist(), errors


# ============================================================================
# PLAN CONSTRUCTION
# ============================================================================

def build_optimized_plan(
    candidates: List[Dict],
    target_calories: float,
    target_protein: float,
    target_carbs: Optional[float] = None,
    target_fat: Optional[float] = None,
    banned_keywords: List[str] = None,
    min_protein_sources: int = 2,
) -> Tuple[Dict, Dict]:
    """
    Build an optimized meal plan from candidates.
    
    1. Select diverse meals (ensuring protein source variety)
    2. Optimize quantities to hit macro targets
    3. Format the plan
    
    Returns:
        plan: The meal plan dict
        metadata: Info about optimization and diversity results
    """
    # Select diverse meals with protein variety requirement
    selected, diversity_info = select_diverse_meals(
        candidates, 
        num_meals=NUM_MEALS_TARGET,
        min_protein_sources=min_protein_sources,
        banned_keywords=banned_keywords,
    )
    
    if not selected:
        return {"meals": []}, {"error": "diversity_selection_failed", "diversity_info": diversity_info}
    
    if len(selected) < 3:
        return {"meals": []}, {"error": "too_few_meals_after_filtering", "diversity_info": diversity_info}
    
    # Optimize quantities
    quantities, errors = optimize_quantities(
        selected,
        target_calories,
        target_protein,
        target_carbs,
        target_fat,
    )
    
    # Build the plan
    meals = []
    for idx, (meal, qty) in enumerate(zip(selected, quantities), start=1):
        base_cal = float(meal.get("calories", 0))
        base_prot = float(meal.get("protein", 0))
        base_carb = float(meal.get("carbs", 0))
        base_fat = float(meal.get("fat", 0))
        
        meals.append({
            "name": str(meal.get("name", "")),
            "quantity": qty,
            "calories": round(base_cal * qty, 1),
            "proteins": round(base_prot * qty, 1),
            "carbs": round(base_carb * qty, 1),
            "fats": round(base_fat * qty, 1),
            "sequence": idx,
        })
    
    plan = {"meals": meals}
    
    # Calculate actual totals for metadata
    total_cal = sum(m["calories"] for m in meals)
    total_prot = sum(m["proteins"] for m in meals)
    total_carb = sum(m["carbs"] for m in meals)
    total_fat = sum(m["fats"] for m in meals)
    
    metadata = {
        "optimization_errors": errors,
        "actual_totals": {
            "calories": total_cal,
            "protein": total_prot,
            "carbs": total_carb,
            "fat": total_fat,
        },
        "num_meals": len(meals),
        "quantities": quantities,
        "diversity_info": diversity_info,  # Include protein diversity info
    }
    
    return plan, metadata


# ============================================================================
# SFT DATA FORMATTING
# ============================================================================

def format_sft_entry(
    system_prompt: str,
    user_content: str,
    search_args: Dict,
    search_results: List[Dict],
    final_plan: Dict,
    include_reasoning: bool = True,
) -> Dict:
    """
    Format a complete SFT training example.
    
    The format follows the tool-calling convention:
    1. System message
    2. User message
    3. Assistant thinks and calls recipe_semantic_search
    4. Tool returns results
    5. Assistant reasons about results and calls return_final_answer_tool
    6. Tool returns the final plan
    """
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # Tool Call 1: Search
    conversation.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "type": "function",
            "function": {
                "name": "recipe_semantic_search",
                "arguments": json.dumps(search_args)
            }
        }]
    })
    
    # Tool Output 1
    conversation.append({
        "role": "tool",
        "name": "recipe_semantic_search",
        "content": json.dumps(search_results)
    })
    
    # Optional: Add reasoning before final answer
    if include_reasoning:
        # Calculate what the model should "reason" about
        total_cal = sum(m["calories"] for m in final_plan.get("meals", []))
        total_prot = sum(m["proteins"] for m in final_plan.get("meals", []))
        reasoning = (
            f"Based on the search results, I'll select diverse meals and optimize quantities "
            f"to hit the macro targets. The plan achieves approximately {total_cal:.0f} kcal "
            f"and {total_prot:.0f}g protein."
        )
        conversation.append({
            "role": "assistant",
            "content": reasoning,
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "return_final_answer_tool",
                    "arguments": json.dumps({"answer": final_plan})
                }
            }]
        })
    else:
        # Direct tool call without reasoning
        conversation.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "return_final_answer_tool",
                    "arguments": json.dumps({"answer": final_plan})
                }
            }]
        })
    
    # Tool Output 2 (Final)
    conversation.append({
        "role": "tool",
        "name": "return_final_answer_tool",
        "content": json.dumps(final_plan)
    })
    
    return {"messages": conversation}


# ============================================================================
# MAIN GENERATION PIPELINE
# ============================================================================

def generate_high_quality_trajectory(scenario: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[float]]:
    """
    Generate a high-quality SFT trajectory for a scenario.
    
    Returns:
        sft_entry: The formatted SFT example, or None if failed
        reward: The calculated reward for the plan
    """
    context = scenario.get("context", {}) or {}
    
    # Extract targets
    target_calories = context.get("daily_cal_target", 2000)
    target_protein = context.get("daily_prot_target", 150)
    target_carbs = context.get("daily_carb_target")
    target_fat = context.get("daily_fat_target")
    banned_keywords = context.get("banned_keywords", [])
    dietary_prefs = context.get("dietary_prefs", "")
    goal = context.get("goal", "")
    
    # Prepare user input
    filtered_context = {
        "daily_cal_target": target_calories,
        "daily_prot_target": target_protein,
        "daily_carb_target": target_carbs,
        "daily_fat_target": target_fat,
        "banned_keywords": banned_keywords,
        "dietary_prefs": dietary_prefs,
        "activity": context.get("activity"),
        "goal": goal,
    }
    
    question = str(scenario.get("input_question", "")).strip()
    context_str = json.dumps(filtered_context, default=str, separators=(",", ":"))
    context_str = _truncate_text(context_str, MAX_CONTEXT_CHARS)
    question = _truncate_text(question, MAX_INPUT_CHARS // 2)
    full_input = f"User Profile/Context:\n{context_str}\n\nRequest: {question}"
    
    # Search for recipes - use diverse protein-focused queries
    # This ensures we get candidates from different protein categories
    base_queries = [
        f"{dietary_prefs} chicken meal {goal}",  # Chicken
        f"{dietary_prefs} beef steak meal",       # Beef
        f"{dietary_prefs} salmon fish meal",      # Fish
        f"{dietary_prefs} tofu chickpea plant protein",  # Plant-based
        f"healthy {dietary_prefs} breakfast oats protein",  # Breakfast
        f"{dietary_prefs} {goal} high protein balanced",    # General
    ]
    
    # Filter out queries that would match banned items
    banned_lower = {str(k).lower() for k in banned_keywords}
    queries = []
    for q in base_queries:
        # Skip query if it contains banned keywords
        if not any(bad in q.lower() for bad in banned_lower):
            queries.append(q)
    
    # Ensure we have at least some queries
    if len(queries) < 2:
        queries = [f"{dietary_prefs} {goal} healthy meal", f"balanced nutrition {dietary_prefs}"]
    
    all_candidates = []
    search_args = {"meal_query": queries[0], "k": 5}  # Primary query for SFT format
    
    for query in queries:
        try:
            results = search_recipes_direct(query.strip(), k=5)
            if isinstance(results, list):
                all_candidates.extend(results)
        except Exception as e:
            logger.warning(f"Search failed for query '{query}': {e}")
    
    # Deduplicate by name
    seen_names = set()
    unique_candidates = []
    for meal in all_candidates:
        name = meal.get("name", "")
        if name and name not in seen_names:
            seen_names.add(name)
            unique_candidates.append(meal)
    
    if len(unique_candidates) < 3:
        logger.warning(f"Not enough candidates: {len(unique_candidates)}")
        return None, None
    
    # Build optimized plan
    plan, metadata = build_optimized_plan(
        unique_candidates,
        target_calories,
        target_protein,
        target_carbs,
        target_fat,
        banned_keywords,
    )
    
    if not plan.get("meals"):
        logger.warning(f"Failed to build plan: {metadata}")
        return None, None
    
    # Validate with reward function
    scenario_data = Scenario(
        id=str(scenario.get("id", "unknown")),
        question=question,
        split="train",
        daily_cal_target=target_calories,
        daily_prot_target=target_protein,
        daily_carb_target=target_carbs,
        daily_fat_target=target_fat,
        banned_keywords=banned_keywords,
    )
    
    # Calculate reward with strict variety checking
    # Our diversity-aware selection ensures protein variety, but we validate with reward function
    try:
        reward, reward_info = combined_reward_v2(
            plan, 
            scenario_data, 
            traj=None,
            skip_llm_judge=True,   # Skip LLM for faster batch processing
            verbose=False,         # Reduce noise during batch generation
            strict_variety=True,   # Require protein source diversity
        )
    except Exception as e:
        logger.warning(f"Reward calculation failed: {e}")
        reward = 0.0
        reward_info = {"error": str(e)}
    
    logger.debug(f"Plan reward: {reward:.3f}, info: {reward_info}")
    
    # Format SFT entry
    sft_entry = format_sft_entry(
        system_prompt=PLANNER_PROMPT,
        user_content=full_input,
        search_args=search_args,
        search_results=unique_candidates[:6],  # Return top 6 as "search results"
        final_plan=plan,
        include_reasoning=True,
    )
    
    return sft_entry, reward


def main():
    """Main generation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate high-quality SFT data for nutrition agent")
    parser.add_argument("--input", type=str, default="../../data/fitness_scenarios_train.parquet",
                        help="Input parquet file path")
    parser.add_argument("--output", type=str, default="../../data/nutrition_sft_data_optimized.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--min-reward", type=float, default=MIN_REWARD_THRESHOLD,
                        help=f"Minimum reward threshold (default: {MIN_REWARD_THRESHOLD})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of scenarios to process")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Resolve paths
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input
    output_path = script_dir / args.output
    
    # Load data
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    df = pd.read_parquet(input_path)
    scenarios = df.to_dict(orient="records")
    
    if args.limit:
        scenarios = scenarios[:args.limit]
    
    logger.info(f"Processing {len(scenarios)} scenarios...")
    
    sft_data = []
    rewards = []
    skipped = 0
    
    for i, scenario in enumerate(scenarios):
        if i % 50 == 0:
            logger.info(f"Progress: {i}/{len(scenarios)} | Accepted: {len(sft_data)} | Skipped: {skipped}")
        
        try:
            sft_entry, reward = generate_high_quality_trajectory(scenario)
            
            if sft_entry is None:
                skipped += 1
                continue
            
            if reward is None or reward < args.min_reward:
                logger.debug(f"Scenario {scenario.get('id')} reward too low: {reward}")
                skipped += 1
                continue
            
            sft_data.append(sft_entry)
            rewards.append(reward)
            
        except Exception as e:
            logger.warning(f"Error processing scenario {scenario.get('id')}: {e}")
            skipped += 1
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in sft_data:
            f.write(json.dumps(entry) + "\n")
    
    # Summary statistics
    logger.info(f"\n{'='*50}")
    logger.info(f"GENERATION COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"Total scenarios: {len(scenarios)}")
    logger.info(f"Accepted (reward >= {args.min_reward}): {len(sft_data)}")
    logger.info(f"Skipped: {skipped}")
    if rewards:
        logger.info(f"Mean reward: {np.mean(rewards):.3f}")
        logger.info(f"Min reward: {np.min(rewards):.3f}")
        logger.info(f"Max reward: {np.max(rewards):.3f}")
    logger.info(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
