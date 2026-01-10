import langgraph
from pydantic import BaseModel
from datetime import datetime
import json
from typing import Any
from functools import wraps

from src.env.nutrition.verifiers_utils import get_payload

from src.nutrition.data_utils import Scenario

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
        # nonlocal final_answer
        payload = get_payload(answer)          # <-- normalize here
        final_answer = FinalAnswer(answer=payload)
        return final_answer.model_dump()


NutritionTools = [recipe_semantic_search, return_final_answer_tool]