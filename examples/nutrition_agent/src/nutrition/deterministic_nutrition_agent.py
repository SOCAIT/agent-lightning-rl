from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, cast

import agentlightning as agl
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nutrition_agent import (  # noqa: E402
    MAX_CONTEXT_CHARS,
    MAX_INPUT_CHARS,
    MockTraj,
    _truncate_text,
)
import src.nutrition.nutrition_tools as nutrition_tools_module  # noqa: E402
from src.env.nutrition.verifiable_rewards.nutrition_rewards import combined_reward_v2  # noqa: E402
from src.nutrition.data_utils import Scenario  # noqa: E402

logger = logging.getLogger(__name__)


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


def _extract_json(content: str) -> dict | None:
    try:
        return json.loads(content)
    except Exception:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


def _render_plan_with_llm(
    model_name: str,
    endpoint: str,
    api_key: str | None,
    temperature: float,
    plan: dict,
    targets: dict,
    allow_optimize: bool,
    handler,
) -> dict:
    policy = (
        "You may adjust quantities to better match the targets."
        if allow_optimize
        else "Do NOT change quantities or macros; only return the plan as JSON."
    )
    prompt = (
        "Return ONLY JSON for a single-day meal plan with schema: "
        '{"meals":[{"name":string,"quantity":number,"calories":number,'
        '"proteins":number,"carbs":number,"fats":number,"sequence":number}]}. '
        f"{policy}"
    )
    payload = json.dumps({"targets": targets, "plan": plan}, separators=(",", ":"))
    model = init_chat_model(
        model_name,
        model_provider="openai",
        openai_api_base=endpoint,
        openai_api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
        temperature=temperature,
        max_retries=0,
        max_tokens=512,
    )
    result = model.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=payload)],
        {"callbacks": [handler] if handler else []},
    )
    if not isinstance(result.content, str):
        return plan
    parsed = _extract_json(result.content)
    return parsed if isinstance(parsed, dict) else plan


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
    if not isinstance(result.content, str):
        return {}
    parsed = _extract_json(result.content)
    return parsed if isinstance(parsed, dict) else {}


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
        context_str = json.dumps(filtered_context, default=str, separators=(",", ":"))
        context_str = _truncate_text(context_str, MAX_CONTEXT_CHARS)
        safe_question = _truncate_text(question, MAX_INPUT_CHARS // 2)
        full_input = f"User Profile/Context:\n{context_str}\n\nRequest: {safe_question}"
        full_input = _truncate_text(full_input, MAX_INPUT_CHARS)

        query = f"{context.get('dietary_prefs','')} {context.get('goal','')} one-day meal plan"
        candidates = nutrition_tools_module.recipe_semantic_search(query.strip(), k=6)
        candidates = _filter_candidates(candidates, banned_keywords)
        plan = _build_plan_from_candidates(candidates, target_calories)

        handler = self.tracer.get_langchain_handler()
        temp = self.val_temperature if self.val_temperature is not None else 0.0

        if plan.get("meals"):
            plan = _render_plan_with_llm(
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
                allow_optimize=self.optimize_with_llm,
                handler=handler,
            )
        else:
            _ = _render_plan_with_llm(
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
                allow_optimize=False,
                handler=handler,
            )

        if self.optimize_with_llm and plan.get("meals"):
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
