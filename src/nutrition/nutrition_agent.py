from __future__ import annotations

import logging
import os
import sys
import json
import time
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, cast, Annotated, Sequence, TypedDict

from dotenv import load_dotenv
load_dotenv()

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

import pandas as pd
import agentlightning as agl

# Import tools and rewards
from src.nutrition.nutrition_tools import NutritionTools
import src.nutrition.nutrition_tools as nutrition_tools_module
from src.env.nutrition.verifiable_rewards.nutrition_rewards import combined_reward_v2
from src.nutrition.data_utils import Scenario

agl.setup_logging(apply_to=[__name__])
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_TURNS = 20 

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
    messages: Annotated[Sequence[BaseMessage], add_messages]

class MockTraj:
    def __init__(self):
        self.messages_and_choices = []

def build_nutrition_agent_system(model_name: str | None = None, endpoint: str | None = None, api_key: str | None = None, temperature: float = 0.2):
    # Initialize the chat model
    # If parameters are provided, use them; otherwise default to existing constants/env
    model = model_name or MODEL_NAME
    
    # We use init_chat_model for flexibility
    # If endpoint/api_key are provided, we assume OpenAI-compatible
    kwargs = {}
    if endpoint:
        kwargs["openai_api_base"] = endpoint
        kwargs["model_provider"] = "openai"
    if api_key:
        kwargs["openai_api_key"] = api_key
    
    chat_model = init_chat_model(model, temperature=temperature, **kwargs)

    # Create the LangGraph ReAct agent
    react_agent = create_react_agent(chat_model, NutritionTools)
    # print("LangGraph agent created!")
    
    return react_agent

class LitNutritionAgent(agl.LitAgent[Dict[str, Any]]):

    def __init__(
        self,
        trained_agents: Optional[str] = None,
        val_temperature: Optional[float] = None,
    ) -> None:
        super().__init__(trained_agents=trained_agents)
        self.val_temperature = val_temperature

    def rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        start_time = time.time()
        
        # 1. Setup Resources
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])
        
        # 2. Setup Trajectory for Tool Logging (Provenance Reward)
        # We inject a fresh MockTraj into the tools module for this rollout
        # Note: This assumes single-threaded execution per process or thread-local storage if parallel
        # Since agl usually runs workers in separate processes, this module-level injection should be safe
        traj = MockTraj()
        nutrition_tools_module.traj = traj

        # 3. Setup Agent
        # Determine temperature
        temp = self.val_temperature if self.val_temperature is not None else llm.sampling_parameters.get("temperature", 0.2)
        
        agent = build_nutrition_agent_system(
            model_name=llm.model,
            endpoint=llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),
            api_key=os.environ.get("OPENAI_API_KEY", "dummy"), # agl might handle auth via proxy, but we pass dummy or env
            temperature=temp
        )

        # 4. Prepare Input
        question = task["input_question"]
        context = task.get("context", {})
        
        # Add context to the prompt or system message? 
        # The PLANNER_PROMPT is static. We can prepend context to the user message.
        # Construct a rich user message with context
        context_str = json.dumps(context, indent=2)
        full_input = f"User Profile/Context:\n{context_str}\n\nRequest: {question}"
        
        logger.info(f"[Rollout {rollout.rollout_id}] Question: {question}")

        # 5. Invoke Agent
        try:
            # We use a large recursion limit to allow for many tool calls
            # Manually prepend the system prompt since state_modifier might not be supported in installed version
            final_state = agent.invoke(
                {"messages": [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=full_input)]},
                {"recursion_limit": MAX_TURNS + 5}
            )
        except Exception as e:
            logger.exception(f"[Rollout {rollout.rollout_id}] Error during agent invocation: {e}")
            return 0.0 # Return 0 reward on failure

        # 6. Extract Result
        # We look for the last tool call to 'return_final_answer_tool' or the final message
        messages = final_state["messages"]
        final_payload = None
        
        # Strategy: Iterate backwards to find the tool call or the tool output
        # The tool 'return_final_answer_tool' returns the JSON dict directly.
        # LangGraph stores tool outputs in ToolMessage.
        
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and msg.name == "return_final_answer_tool":
                try:
                    # The tool returns a dict (result of model_dump)
                    # ToolMessage content is usually string. 
                    # If create_react_agent stores artifact, check that.
                    # But standard ToolMessage content is string.
                    # Our tool returns a dict, so LangChain/LangGraph might json.dumps it.
                    content = msg.content
                    if isinstance(content, str):
                        final_payload = json.loads(content)
                    else:
                        final_payload = content
                    break
                except Exception:
                    logger.warning(f"[Rollout {rollout.rollout_id}] Failed to parse tool output")
                    continue
        
        if final_payload is None:
            logger.warning(f"[Rollout {rollout.rollout_id}] No final answer tool call found.")
            return 0.0

        # 7. Calculate Reward
        # Extract targets from context
        daily_cal_target = context.get("daily_cal_target", 2000)
        daily_prot_target = context.get("daily_prot_target", 150)
        daily_carb_target = context.get("daily_carb_target")
        daily_fat_target = context.get("daily_fat_target")
        banned_keywords = context.get("banned_keywords", [])
        
        # Create Scenario Object
        scenario_data = Scenario(
            id=str(task.get("id", "unknown")),
            question=question,
            split=task.get("split", "test"),
            daily_cal_target=daily_cal_target,
            daily_prot_target=daily_prot_target,
            daily_carb_target=daily_carb_target,
            daily_fat_target=daily_fat_target,
            banned_keywords=banned_keywords
        )

        # Calculate Combined Reward
        try:
             final_reward, info = asyncio.run(combined_reward_v2(final_payload, scenario_data, traj))
        except Exception as e:
            logger.exception(f"[Rollout {rollout.rollout_id}] Error calculating reward: {e}")
            final_reward = 0.0
            info = {"error": str(e)}

        logger.info(f"[Rollout {rollout.rollout_id}] Final Reward: {final_reward:.2f}")
        logger.info(f"[Rollout {rollout.rollout_id}] Info: {info}")
        
        end_time = time.time()
        logger.info(f"[Rollout {rollout.rollout_id}] Time: {end_time - start_time:.2f}s")
        
        return final_reward

def debug_nutrition_agent():
    # Load some data
    data_path = os.path.join(project_root, "data", "fitness_scenarios.jsonl")
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return

    # Read first 5 lines
    with open(data_path, "r") as f:
        lines = [json.loads(line) for line in f.readlines()[:5]]
    
    trainer = agl.Trainer(
        n_workers=1,
        initial_resources={
            "main_llm": agl.LLM(
                endpoint=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
                model="gpt-3.5-turbo", # Use a default or valid model
                sampling_parameters={"temperature": 0.2},
            )
        },
    )
    
    # Run dev
    trainer.dev(LitNutritionAgent(), lines)

if __name__ == "__main__":
    debug_nutrition_agent()
