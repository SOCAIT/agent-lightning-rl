import langgraph
from pydantic import BaseModel, field_validator
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from dataclasses import asdict, dataclass, is_dataclass

from datetime import datetime
import json
from functools import wraps

from src.env.nutrition.verifiers_utils import get_payload

from src.nutrition.data_utils import Scenario
# Pinecone
from pinecone import Pinecone
import os
# LangChain and LangGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ============================================================================
# PINECONE SETUP
# ============================================================================

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

recipe_index_name = "syntrafit-meals-nutrition"
exercise_index_name = "syntrafit-exercises"

recipe_index = pc.Index(recipe_index_name)
exercise_index = pc.Index(exercise_index_name)

def extract_meal_names(data):
    data = data['result']
    return_data = []


    return [{"id" : hit['_id'] ,"name": hit["fields"]["name"], "calories":  hit["fields"]["calories"],"carbs":  hit["fields"]["carbs"], "protein": hit["fields"]["proteins"], "fat":  hit["fields"]["fats"]}  for hit in data["hits"] if "fields" in hit and "name" in hit["fields"]]

 # Search the dense index
results = recipe_index.search(
          namespace="syntrafit",
          query={
              "top_k": 2,
              "inputs": {
                  'text': " Chicken and rice healthy"
              }
          },
          rerank={
          "model": "bge-reranker-v2-m3",
          "top_n": 2,
          "rank_fields": [ "name"]
    },
      )

print(results)

extract_meal_names(results)


class FinalAnswer(BaseModel):
    answer: Dict[str, Any]

    @field_validator("answer", mode="before")
    @classmethod
    def ensure_dict(cls, v):
            # Unwrap nested FinalAnswer by mistake
            if isinstance(v, FinalAnswer):
                return v.answer
            # Parse JSON string
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except json.JSONDecodeError as e:
                    raise ValueError(f"answer must be a JSON object string or dict; got invalid JSON: {e}")
            # Already a dict
            if isinstance(v, dict):
                return v
            raise TypeError(f"Unsupported type for answer: {type(v).__name__}")


@dataclass
class SearchResult:
    message_id: str
    snippet: str



class NutritionTrajectory(BaseModel):
    final_answer: FinalAnswer | None = None

class FitnessScenario(BaseModel):
    step: int
    scenario: Scenario

 
 
def log_tool(tool_name):
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                GREEN = "\033[92m"
                RESET = "\033[0m"

                call = {
                    "tool": tool_name,
                    "ts": datetime.utcnow().isoformat(),
                    "args": args,
                    "kwargs": kwargs
                }

                color_prefix = GREEN if "final_answer" in tool_name else ""
                color_reset = RESET if "final_answer" in tool_name else ""

                print(f"{color_prefix}[TOOL START] {tool_name} args={kwargs}{color_reset}")
                traj.messages_and_choices.append(
                    {"role": "tool_log", "content": json.dumps({"start": call})}
                )

                try:
                    out = fn(*args, **kwargs)
                    print(f"{color_prefix}[TOOL END] {tool_name} result_preview={str(out)[:400]}{color_reset}")
                    traj.messages_and_choices.append(
                        {"role": "tool_log", "content": json.dumps({"end": {**call, 'result': out}})}
                    )
                    return out
                except Exception as e:
                    print(f"\033[91m[TOOL ERROR] {tool_name}: {e}\033[0m")
                    traj.messages_and_choices.append(
                        {"role": "tool_log", "content": json.dumps({"error": {**call, "error": str(e)}})}
                    )
                    raise
            return wrapper
        return decorator

def _normalize_meal_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
    print(f"DEBUG: Normalizing payload keys: {list(payload.keys())}")
    meals: List[Dict[str, Any]] | None = None
    if isinstance(payload.get("meals"), list):
        meals = payload.get("meals")
    elif isinstance(payload.get("daily_plan"), list):
        meals = payload.get("daily_plan")
    elif isinstance(payload.get("dailyPlan"), list):
        meals = payload.get("dailyPlan")
    elif isinstance(payload.get("dailyMealPlans"), list) and payload.get("dailyMealPlans"):
        first_day = payload.get("dailyMealPlans")[0]
        if isinstance(first_day, dict):
            meals = first_day.get("meals")

    if not meals:
        return payload

    normalized: List[Dict[str, Any]] = []
    seq = 1

    for meal in meals:
        if not isinstance(meal, dict):
            continue
        
        # Flatten nested 'meal' dicts or handle 'meal' as string name
        candidates = meal.copy()
        if isinstance(meal.get("meal"), dict):
            candidates.update(meal.get("meal"))
        elif isinstance(meal.get("meal"), str):
            candidates["name"] = meal.get("meal")
            
        items = meal.get("items")
        if isinstance(items, list) and items:
            for item in items:
                if not isinstance(item, dict):
                    continue
                # Merge item with parent candidates to inherit missing fields if needed
                item_candidates = candidates.copy()
                item_candidates.update(item)
                
                normalized.append(
                    {
                        "name": item_candidates.get("name") or item_candidates.get("meal") or item_candidates.get("meal_name") or "Unknown Meal",
                        "quantity": float(item_candidates.get("quantity", 1)),
                        "calories": float(item_candidates.get("calories", 0)),
                        "proteins": float(item_candidates.get("proteins", item_candidates.get("protein", 0))),
                        "carbs": float(item_candidates.get("carbs", item_candidates.get("carb", 0))),
                        "fats": float(item_candidates.get("fats", item_candidates.get("fat", 0))),
                        "sequence": int(item_candidates.get("sequence", seq)),
                    }
                )
                seq += 1
            continue

        normalized.append(
            {
                "name": candidates.get("name") or candidates.get("meal") or candidates.get("meal_name") or "Unknown Meal",
                "quantity": float(candidates.get("quantity", 1)),
                "calories": float(candidates.get("calories", 0)),
                "proteins": float(candidates.get("proteins", candidates.get("protein", 0))),
                "carbs": float(candidates.get("carbs", candidates.get("carb", 0))),
                "fats": float(candidates.get("fats", candidates.get("fat", 0))),
                "sequence": int(candidates.get("sequence", seq)),
            }
        )
        seq += 1

    payload["meals"] = normalized
    print(f"DEBUG: Normalized {len(normalized)} meals.")
    return payload
@tool
@log_tool("recipe_semantic_search")
def recipe_semantic_search(meal_query: str, k: int = 5) -> str:
      """Search the recipe index for the most similar recipes to the query."""
      # Hard cap to keep tool output small for context limits
      k = min(max(int(k), 1), 3)
      # Search the dense index
      results = recipe_index.search(
          namespace="syntrafit",
          query={
              "top_k": k,
              "inputs": {
                  'text': meal_query
              }
          }
      )

      results = extract_meal_names(results)
      # Trim verbose fields defensively
      trimmed = []
      for item in results[:k]:
          trimmed.append(
              {
                  "name": str(item.get("name", ""))[:80],
                  "calories": float(item.get("calories", 0)),
                  "carbs": float(item.get("carbs", 0)),
                  "protein": float(item.get("protein", 0)),
                  "fat": float(item.get("fat", 0)),
              }
          )

      print(trimmed)

      return trimmed


@tool
@log_tool("return_final_answer_tool")
def return_final_answer_tool(answer: Dict[str, Any]) -> dict:
        """Return the final answer (daily meal plan) in the correct format."""
        payload = get_payload(answer)  # Normalize in case of schema drift
        if isinstance(payload, dict):
            payload = _normalize_meal_plan(payload)
        final_answer = FinalAnswer(answer=payload)
        return final_answer.model_dump()


NutritionTools = [recipe_semantic_search, return_final_answer_tool]