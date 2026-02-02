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

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
# Alternatives:
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Smaller, faster
# MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"  # Math-focused but worse at tool calling
MAX_TURNS = 3  # Reduced to prevent context overflow
MAX_CONTEXT_CHARS = 300  # Reduced for shorter prompts
MAX_INPUT_CHARS = 600  # Reduced for shorter prompts

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
- Tool call arguments MUST be valid JSON. The tool expects an object, not a JSON string.
- Call format: return_final_answer_tool({{"answer": {{"meals": [ ... ]}}}})
- JSON schema: {{"meals":[{{"name":str,"quantity":number,"calories":number,"proteins":number,"carbs":number,"fats":number,"sequence":int}}]}}
- Do NOT nest meals under "items" or change key names (e.g. "protein" -> "proteins").
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

def _build_fallback_plan(context: dict) -> dict:
    banned_keywords = {str(k).lower() for k in (context.get("banned_keywords") or [])}
    search_query = "balanced one-day meal plan"
    results = nutrition_tools_module.recipe_semantic_search.invoke(
        {"meal_query": search_query, "k": 5}
    )
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

def _log_message_tail(rollout_id: str, messages: Sequence[BaseMessage], limit: int = 6) -> None:
    if not messages:
        logger.info(f"[Rollout {rollout_id}] No messages to log.")
        return
    tail = list(messages)[-limit:]
    for idx, msg in enumerate(tail, start=1):
        msg_type = msg.__class__.__name__
        msg_name = getattr(msg, "name", None)
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            try:
                content = json.dumps(content, default=str)
            except Exception:
                content = str(content)
        content = _truncate_text(content, 500)
        name_str = f" name={msg_name}" if msg_name else ""
        logger.info(f"[Rollout {rollout_id}] Message {idx}/{len(tail)} {msg_type}{name_str}: {content}")

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
        strict_failures: bool = False,
        debug_messages: bool = False,
        allow_fallback_plan: bool = True,
    ) -> None:
        super().__init__(trained_agents=trained_agents)
        self.val_temperature = val_temperature
        self.strict_failures = strict_failures
        self.debug_messages = debug_messages
        self.allow_fallback_plan = allow_fallback_plan

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
            if self.strict_failures:
                raise
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
            logger.warning(f"[Rollout {rollout.rollout_id}] No final payload parsed.")
            if self.debug_messages:
                _log_message_tail(rollout.rollout_id, messages)
            if self.strict_failures:
                raise RuntimeError("No final payload parsed from agent output.")
            if self.allow_fallback_plan:
                final_payload = _build_fallback_plan(context)
            else:
                return None

        # Normalize payload aggressively before reward calculation
        if isinstance(final_payload, dict):
            try:
                final_payload = nutrition_tools_module._normalize_meal_plan(final_payload)
            except Exception as e:
                logger.warning(f"[Rollout {rollout.rollout_id}] Normalization failed: {e}")

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
             logger.info(f"Combined reward v2: {final_payload}, {scenario_data}, {traj}")
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
