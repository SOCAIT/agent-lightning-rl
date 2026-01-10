import langgraph
from pydantic import BaseModel
from datetime import datetime
import json
from typing import Any
from functools import wraps



class NutritionTrajectory(langgraph.Trajectory):
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
@tool
@log_tool("recipe_semantic_search")
def recipe_semantic_search(meal_query: str, k: int = 5) -> str:
      """Search the recipe index for the most similar recipes to the query."""
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

      print(results)

      return results


@tool
@log_tool("return_final_answer_tool")
def return_final_answer_tool(answer: str) -> dict:
        """Return the final answer (daily meal plan) in the correct format """
        nonlocal final_answer
        payload = get_payload(answer)          # <-- normalize here
        final_answer = FinalAnswer(answer=payload)
        return final_answer.model_dump()


NutritionTools = [recipe_semantic_search, return_final_answer_tool]