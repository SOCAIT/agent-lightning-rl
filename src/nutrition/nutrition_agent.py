from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from langchain_core.tools import tool

from src.nutrition.nutrition_tools import NutritionTools

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

PLANNER_PROMPT = f"""
You are a nutrition planner specialist who creates daily nutrition plans. Think carefully and pay great attention to macro numbers.

You must create a one-day meal plan that meets the user's macro and dietary targets.

TOOLS YOU CAN USE (names must match exactly):
1) recipe_semantic_search(meal_query, k) – Search for relevant recipes and return their true macros.
2) return_final_answer_tool(answer) – Return the final answer (JSON plan).

ROUTING RULES (VERY IMPORTANT):
- If the user asks for a meal/nutrition plan, you MUST NOT call get_available_exercises or generate_workout_plan_mutation.
- Even if the request is for a 7-day plan, create only a single-day meal plan.

EXECUTION POLICY:
- You MAY call tools during reasoning to gather information and choose recipes.
- The FINAL assistant message must output ONLY a single call to return_final_answer_tool with the exact JSON plan (stringified if needed).
- You may take up to {MAX_TURNS} turns to find the answer.

============================================================
NUTRITION PLAN PIPELINE
============================================================

1) PLAN SKELETON
   • Generate a one-day meal plan for the user. The plan should have meals that fulfill the user's daily macro targets.
   • Create a reasonable number of meals (meals can include snacks). Base the count/portions on the user's macro targets.
   • Meal names in the final plan MUST be recipe names returned by recipe_semantic_search (do not invent names).

2) USE RECIPE SEMANTIC SEARCH TOOL
   • Use recipe_semantic_search to find relevant recipes with correct nutrition info.
   • You MUST ALWAYS use the macros retrieved from the tool and NOT infer your own data.
   • For each meal you include:
     - Set "name" to the exact recipe name from the tool
     - Set "quantity" to a multiplier (e.g., 1.0 for one serving, 1.5 for 1.5 servings, 0.5 for half serving)
     - Calculate macros as: quantity × base_recipe_macros
     - Example: If recipe has 400 cal and quantity=1.5, then calories should be 600
   • If a candidate recipe includes any banned keywords/ingredients from context, discard it and search again.

3) MACRO ADJUSTMENT (per day)
   • Sum macros for the day across all meals.
   • If totals differ from the user's daily targets, adjust the "quantity" field for meals (still using tool macros) until daily totals are within ±5% of the user's targets (calories/protein/carbs/fat).
   • You can use fractional quantities like 0.75, 1.25, 1.5 to hit precise macro targets.
   • Respect ALL banned keywords/ingredients from context.

4) JSON MEAL PLAN (scratch-step)
   • Build JSON matching this schema (no comments):

     {{
       "meals": [
         {{
           "name": "Grilled Chicken & Rice",
           "quantity": 1.5,
           "calories": 700,
           "proteins": 45,
           "carbs": 60,
           "fats": 20,
           "sequence": 1
         }}
       ]
     }}

   • The "quantity" field is a multiplier (can be fractional like 1.5, 0.75, 2.0) that represents portions/servings.
   • Macros (calories, proteins, carbs, fats) should be: quantity × base_recipe_macros from recipe_semantic_search.
   • Ensure the summed macros for the day are within ±5% of the targets.
   • IMPORTANT: Every "name" MUST match a recipe name previously returned by recipe_semantic_search. 
   • The quantity field allows precise macro targeting by scaling recipe portions.

5) IF YOU REACHED MAX_TURNS and you have not found a final answer, return the best final answer you can with all information you gathered.

6) TOOL CALL (FINALIZE)
   • Call return_final_answer_tool with:
     - answer = the EXACT JSON meal plan (stringified if needed)

"""




class AgentState(TypedDict):
    """The state of the agent."""

    # add_messages is a reducer
    # See https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
    messages: Annotated[Sequence[BaseMessage], add_messages]




def build_nutrition_agent_system():
     # Pass the local path to the model
    chat_model = init_chat_model(f"{MODEL_NAME}", temperature=0.2)


    # Create the LangGraph ReAct agent
    react_agent = create_react_agent(chat_model, NutritionTools)
    print("LangGraph agent created!")
    print(react_agent)

    MAX_TURNS = 20 
    return react_agent
