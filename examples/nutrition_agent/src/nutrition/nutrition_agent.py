from __future__ import annotations

import logging
import os
import sys
import json
import time
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
MAX_TURNS = 6
MAX_CONTEXT_CHARS = 400
MAX_INPUT_CHARS = 800
FALLBACK_INPUT_CHARS = 400

PLANNER_PROMPT = f"""
You are a nutrition planner. Create a ONE-DAY meal plan that matches the user's macros and dietary restrictions.

Tools:
- recipe_semantic_search(meal_query, k)
- return_final_answer_tool(answer)

Rules:
- Always return a single-day plan even if user asks for 7 days.
- Meal names MUST come from recipe_semantic_search results.
- Macros must be based on tool results and scaled by quantity (serving multiplier).
- Final response MUST be a single call to return_final_answer_tool with the JSON plan.
"""

class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

class MockTraj:
    def __init__(self):
        self.messages_and_choices = []

def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 12] + " ...(truncated)"

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

def _invoke_fallback_model(
    model_name: str,
    endpoint: str,
    api_key: str | None,
    temperature: float,
    full_input: str,
    handler,
):
    fallback_prompt = (
        "Return ONLY a single-day JSON meal plan with fields: "
        "meals[{name,quantity,calories,proteins,carbs,fats,sequence}]."
    )
    fallback_input = _truncate_text(full_input, FALLBACK_INPUT_CHARS)
    fallback_model = init_chat_model(
        model_name,
        model_provider="openai",
        openai_api_base=endpoint,
        openai_api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
        temperature=temperature,
        max_retries=0,
        max_tokens=512,
    )
    result = fallback_model.invoke(
        [SystemMessage(content=fallback_prompt), HumanMessage(content=fallback_input)],
        {"callbacks": [handler] if handler else []},
    )
    return result

def _build_fallback_plan(context: dict) -> dict:
    banned_keywords = {str(k).lower() for k in (context.get("banned_keywords") or [])}
    search_query = "balanced one-day meal plan"
    results = nutrition_tools_module.recipe_semantic_search(search_query, k=5)
    for meal in results:
        name = str(meal.get("name", ""))
        if any(bad in name.lower() for bad in banned_keywords):
            continue
        calories = float(meal.get("calories", 0))
        carbs = float(meal.get("carbs", 0))
        protein = float(meal.get("protein", 0))
        fat = float(meal.get("fat", 0))
        return {
            "meals": [
                {
                    "name": name,
                    "quantity": 1.0,
                    "calories": calories,
                    "proteins": protein,
                    "carbs": carbs,
                    "fats": fat,
                    "sequence": 1,
                }
            ]
        }
    return {"meals": []}

def _filter_candidates(candidates: list[dict], banned_keywords: list[str]) -> list[dict]:
    banned = {str(k).lower() for k in banned_keywords}
    filtered: list[dict] = []
    for meal in candidates:
        name = str(meal.get("name", "")).strip()
        if not name:
            continue
        if any(bad in name.lower() for bad in banned):
            continue
        filtered.append(meal)
    return filtered

def _build_plan_from_candidates(candidates: list[dict], target_calories: float | None) -> dict:
    if not candidates:
        return {"meals": []}
    selected = candidates[:3]
    base_total = sum(float(m.get("calories", 0)) for m in selected)
    if base_total <= 0:
        scale = 1.0
    elif target_calories and target_calories > 0:
        scale = target_calories / base_total
    else:
        scale = 1.0
    scale = max(0.5, min(2.0, scale))
    meals = []
    for idx, meal in enumerate(selected, start=1):
        calories = float(meal.get("calories", 0)) * scale
        carbs = float(meal.get("carbs", 0)) * scale
        protein = float(meal.get("protein", 0)) * scale
        fat = float(meal.get("fat", 0)) * scale
        meals.append(
            {
                "name": str(meal.get("name", "")),
                "quantity": round(scale, 2),
                "calories": round(calories, 2),
                "proteins": round(protein, 2),
                "carbs": round(carbs, 2),
                "fats": round(fat, 2),
                "sequence": idx,
            }
        )
    return {"meals": meals}

def _optimize_plan_with_llm(
    model_name: str,
    endpoint: str,
    api_key: str | None,
    temperature: float,
    plan: dict,
    targets: dict,
    handler,
) -> dict:
    prompt = (
        "Adjust ONLY the 'quantity' fields to better match the targets. "
        "Do not change meal names. Return JSON ONLY with schema: "
        '{"meals":[{"name":string,"quantity":number}]}.'
    )
    payload = json.dumps({"targets": targets, "plan": plan}, separators=(",", ":"))
    model = init_chat_model(
        model_name,
        model_provider="openai",
        openai_api_base=endpoint,
        openai_api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
        temperature=temperature,
        max_retries=0,
        max_tokens=256,
    )
    result = model.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=payload)],
        {"callbacks": [handler] if handler else []},
    )
    try:
        return json.loads(result.content)
    except Exception:
        return {}

def _apply_quantities(plan: dict, quantities_payload: dict) -> dict:
    meals = plan.get("meals", [])
    updates = {m.get("name"): m.get("quantity") for m in quantities_payload.get("meals", [])}
    for meal in meals:
        name = meal.get("name")
        if name in updates:
            qty = float(updates[name])
            base_qty = float(meal.get("quantity", 1.0))
            if base_qty <= 0:
                base_qty = 1.0
            factor = qty / base_qty
            meal["quantity"] = round(qty, 2)
            meal["calories"] = round(float(meal.get("calories", 0)) * factor, 2)
            meal["proteins"] = round(float(meal.get("proteins", 0)) * factor, 2)
            meal["carbs"] = round(float(meal.get("carbs", 0)) * factor, 2)
            meal["fats"] = round(float(meal.get("fats", 0)) * factor, 2)
    return plan

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
        question = str(task.get("input_question", "")).strip()
        if not question:
            logger.warning(f"[Rollout {rollout.rollout_id}] Empty question. Skipping.")
            return None

        context = task.get("context", {}) or {}
        filtered_context = {
            "daily_cal_target": context.get("daily_cal_target"),
            "daily_prot_target": context.get("daily_prot_target"),
            "daily_carb_target": context.get("daily_carb_target"),
            "daily_fat_target": context.get("daily_fat_target"),
            "banned_keywords": context.get("banned_keywords"),
            "dietary_prefs": context.get("dietary_prefs"),
            "activity": context.get("activity"),
            "goal": context.get("goal"),
        }
        
        # Add context to the prompt or system message? 
        # The PLANNER_PROMPT is static. We can prepend context to the user message.
        # Construct a rich user message with context
        context_str = json.dumps(filtered_context, default=str, separators=(",", ":"))
        context_str = _truncate_text(context_str, MAX_CONTEXT_CHARS)
        question = _truncate_text(question, MAX_INPUT_CHARS // 2)
        full_input = f"User Profile/Context:\n{context_str}\n\nRequest: {question}"
        full_input = _truncate_text(full_input, MAX_INPUT_CHARS)
        
        logger.info(f"[Rollout {rollout.rollout_id}] Question: {question}")

        # 5. Invoke Agent
        try:
            # We use a large recursion limit to allow for many tool calls
            # Manually prepend the system prompt since state_modifier might not be supported in installed version
            handler = self.tracer.get_langchain_handler()
            final_state = agent.invoke(
                {"messages": [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=full_input)]},
                {
                    "recursion_limit": MAX_TURNS + 2,
                    "callbacks": [handler] if handler else [],
                },
            )
        except Exception as e:
            logger.exception(f"[Rollout {rollout.rollout_id}] Error during agent invocation: {e}")
            try:
                fallback_result = _invoke_fallback_model(
                    model_name=llm.model,
                    endpoint=llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),
                    api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
                    temperature=temp,
                    full_input=full_input,
                    handler=handler,
                )
                final_state = {"messages": [fallback_result]}
            except Exception as fallback_error:
                logger.exception(
                    f"[Rollout {rollout.rollout_id}] Fallback model invocation failed: {fallback_error}"
                )
                return None

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
            # Fallback: try to parse the last assistant message as JSON.
            last_msg = messages[-1] if messages else None
            if last_msg is not None and hasattr(last_msg, "content"):
                try:
                    final_payload = json.loads(last_msg.content)  # type: ignore[arg-type]
                except Exception:
                    final_payload = None

        if final_payload is None:
            final_payload = _build_fallback_plan(context)

        # 7. Calculate Reward
        # Extract targets from context
        daily_cal_target = context.get("daily_cal_target", 2000)
        daily_prot_target = context.get("daily_prot_target", 150)
        daily_carb_target = context.get("daily_carb_target")
        daily_fat_target = context.get("daily_fat_target")
        banned_keywords = context.get("banned_keywords", [])
        
        # Create Scenario Object
        split = task.get("split", "test")
        if split == "val":
            split = "test"
        scenario_data = Scenario(
            id=str(task.get("id", "unknown")),
            question=question,
            split=split,
            daily_cal_target=daily_cal_target,
            daily_prot_target=daily_prot_target,
            daily_carb_target=daily_carb_target,
            daily_fat_target=daily_fat_target,
            banned_keywords=banned_keywords
        )

        # Calculate Combined Reward
        try:
             final_reward, info = combined_reward_v2(final_payload, scenario_data, traj)
        except Exception as e:
            logger.exception(f"[Rollout {rollout.rollout_id}] Error calculating reward: {e}")
            final_reward = 0.0
            info = {"error": str(e)}

        logger.info(f"[Rollout {rollout.rollout_id}] Final Reward: {final_reward:.2f}")
        logger.info(f"[Rollout {rollout.rollout_id}] Info: {info}")
        
        end_time = time.time()
        logger.info(f"[Rollout {rollout.rollout_id}] Time: {end_time - start_time:.2f}s")
        
        return final_reward

class LitNutritionAgentDeterministic(agl.LitAgent[Dict[str, Any]]):
    def __init__(
        self,
        trained_agents: Optional[str] = None,
        optimize_with_llm: bool = False,
        val_temperature: Optional[float] = None,
    ) -> None:
        super().__init__(trained_agents=trained_agents)
        self.optimize_with_llm = optimize_with_llm
        self.val_temperature = val_temperature

    def rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        start_time = time.time()
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])

        traj = MockTraj()
        nutrition_tools_module.traj = traj

        question = str(task.get("input_question", "")).strip()
        if not question:
            logger.warning(f"[Rollout {rollout.rollout_id}] Empty question. Skipping.")
            return None

        context = task.get("context", {}) or {}
        banned_keywords = context.get("banned_keywords") or []
        target_calories = context.get("daily_cal_target")

        query = f"{context.get('dietary_prefs','')} {context.get('goal','')} one-day meal plan"
        candidates = nutrition_tools_module.recipe_semantic_search(query.strip(), k=6)
        candidates = _filter_candidates(candidates, banned_keywords)
        plan = _build_plan_from_candidates(candidates, target_calories)

        if self.optimize_with_llm and plan.get("meals"):
            handler = self.tracer.get_langchain_handler()
            temp = self.val_temperature if self.val_temperature is not None else 0.0
            quantities_payload = _optimize_plan_with_llm(
                model_name=llm.model,
                endpoint=llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),
                api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
                temperature=temp,
                plan=plan,
                targets={
                    "calories": context.get("daily_cal_target"),
                    "protein": context.get("daily_prot_target"),
                    "carbs": context.get("daily_carb_target"),
                    "fat": context.get("daily_fat_target"),
                },
                handler=handler,
            )
            plan = _apply_quantities(plan, quantities_payload)

        split = task.get("split", "test")
        if split == "val":
            split = "test"
        scenario_data = Scenario(
            id=str(task.get("id", "unknown")),
            question=question,
            split=split,
            daily_cal_target=context.get("daily_cal_target", 2000),
            daily_prot_target=context.get("daily_prot_target", 150),
            daily_carb_target=context.get("daily_carb_target"),
            daily_fat_target=context.get("daily_fat_target"),
            banned_keywords=banned_keywords,
        )

        final_reward, info = combined_reward_v2(plan, scenario_data, traj)
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
