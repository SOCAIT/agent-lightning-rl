from __future__ import annotations

import argparse
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import ray
from nutrition_agent import LitNutritionAgent
from deterministic_nutrition_agent import LitNutritionAgentDeterministic

import agentlightning as agl

RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_files": "data/fitness_scenarios_train.parquet",
        "val_files": "data/fitness_scenarios_val.parquet",
        "train_batch_size": 16,  # Conservative for stability
        "max_prompt_length": 6144,  # Conservative: supports multi-turn tool calls (6 turns × ~1K tokens/turn)
        "max_response_length": 2048,  # Sufficient for complete plan generation
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 2,  # Conservative parallel rollouts
            "log_prob_micro_batch_size_per_gpu": 2,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            "gpu_memory_utilization": 0.5,  # 50% (~70GB for vLLM)
            "max_model_len": 8192,  # Reduced to 8K for stability (prompt + response)
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                    "max_num_seqs": 16,  # Further reduced concurrent sequences
                    "max_num_batched_tokens": 8192,  # Limit batched tokens to prevent KV cache issues
                    "enable_chunked_prefill": False,  # Disable chunked prefill (can cause issues)
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 16,  # Reduced to match train_batch_size
            "ppo_micro_batch_size_per_gpu": 4,
            "optim": {"lr": 1e-6},
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {
                "param_offload": True,
                "optimizer_offload": True,
            },
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 8,  # Increased back to 8 for H200
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            "path": "Qwen/Qwen2.5-7B-Instruct",  # Updated to 7B model
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 1,
        "val_before_train": True,
        "critic_warmup": 0,
        "logger": ["console", "wandb"],
        "project_name": "AgentLightning",
        "experiment_name": "nutrition",
        "nnodes": 1,
        "test_freq": 16,
        "total_epochs": 5,
    },
}


def config_train_fast() -> Dict[str, Any]:
    """A fast training run for CI testing purposes."""

    # `EXPERIMENT_NAME="spider_$(date +%Y%m%d%H%M%S)"`
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    EXPERIMENT_NAME = f"nutrition_{timestamp}"

    # `PROJECT_NAME=AgentLightningCI`
    PROJECT_NAME = "AgentLightningCI"

    # Simulate writing to $GITHUB_OUTPUT if it’s set
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"project_name={PROJECT_NAME}\n")
            f.write(f"run_name={EXPERIMENT_NAME}\n")

    print("Set environment variables:")
    print(f"PROJECT_NAME={PROJECT_NAME}")
    print(f"EXPERIMENT_NAME={EXPERIMENT_NAME}")

    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.6
    config["actor_rollout_ref"]["model"]["path"] = "Qwen/Qwen2.5-7B-Instruct"
    config["data"]["val_files"] = "data/fitness_scenarios_val.parquet"
    config["trainer"]["total_epochs"] = 1
    config["trainer"]["total_training_steps"] = 1
    config["trainer"]["experiment_name"] = EXPERIMENT_NAME
    config["trainer"]["project_name"] = PROJECT_NAME
    config["trainer"]["test_freq"] = 1
    return config


def config_train_qwen() -> Dict[str, Any]:
    """A configuration for training with Qwen-2.5B."""

    config = deepcopy(RL_TRAINING_CONFIG)
    return config


def config_train_npu() -> Dict[str, Any]:
    """A configuration for training with NPU."""

    config = deepcopy(RL_TRAINING_CONFIG)
    # Note: engine_kwargs is commented out in base config for HF backend
    # If using vLLM, uncomment engine_kwargs in base config first
    # del config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["enable_auto_tool_choice"]
    # del config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["tool_call_parser"]
    del config["trainer"]["logger"][1]
    config["actor_rollout_ref"]["actor"]["use_torch_compile"] = False
    config["trainer"]["val_before_train"] = False
    config["trainer"]["save_freq"] = 256
    config["trainer"]["device"] = "npu"
    return config


def config_train_llama() -> Dict[str, Any]:
    """A configuration for training with LLaMA-3.2-1B-Instruct.

    You will need a `HF_TOKEN` set to run with this config.
    Note: This config requires vLLM backend. Uncomment engine_kwargs in base config
    and set rollout name to "vllm" to use this configuration.
    """

    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["rollout"]["multi_turn"]["format"] = "llama3_json"
    # Note: engine_kwargs is commented out in base config for HF backend
    # Uncomment engine_kwargs in base config and set name to "vllm" to use this
    # config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["tool_call_parser"] = "llama3_json"
    config["actor_rollout_ref"]["model"]["path"] = "Qwen/Qwen2.5-1.5B-Instruct"
    return config


def train(
    config: Dict[str, Any],
    active_agent: Optional[str],
    agent_type: str,
    strict_failures: bool,
    debug_agent: bool,
    allow_fallback_plan: bool,
) -> None:
    """Train the SQL agent with the given configuration."""

    if agent_type == "deterministic":
        agent = LitNutritionAgentDeterministic(optimize_with_llm=False)
    elif agent_type == "deterministic-llm":
        agent = LitNutritionAgentDeterministic(optimize_with_llm=True)
    else:
        agent = LitNutritionAgent(
            strict_failures=strict_failures,
            debug_messages=debug_agent,
            allow_fallback_plan=allow_fallback_plan,
        )
    algorithm = agl.VERL(config)
    trainer = agl.Trainer(n_runners=10, algorithm=algorithm, adapter={"agent_match": active_agent})
    print("Adapter agent match acknowledged:", trainer.adapter.agent_match)  # type: ignore

    train_data = pd.read_parquet(config["data"]["train_files"]).to_dict(orient="records")  # type: ignore
    val_data = pd.read_parquet(config["data"]["val_files"]).to_dict(orient="records")  # type: ignore
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)  # type: ignore


def main() -> None:
    """Main function to parse arguments and run training."""
    # Initialize Ray with local mode or just current environment to avoid venv creation issues
    print(f"Ray is installed at: {ray.__file__}")
    if not ray.is_initialized():
        # Force using the current environment and disable complex runtime_env creation
        ray.init(ignore_reinit_error=True, runtime_env=None)

    parser = argparse.ArgumentParser(
        description="Train an SQL agent on the Spider dataset using different model configurations"
    )

    parser.add_argument(
        "config",
        choices=["fast", "qwen", "llama", "npu"],
        help="Training configuration: 'fast' (CI testing), 'qwen' (Qwen-2.5-Coder-1.5B), 'llama' (LLaMA-3.2-3B),'npu' (Train with NPU)",
    )

    parser.add_argument(
        "--active-agent", type=str, help="Override the active agent name (default: auto-generated based on config)"
    )
    parser.add_argument(
        "--agent-type",
        choices=["llm", "deterministic", "deterministic-llm"],
        default="llm",
        help="Agent implementation to use for rollout behavior.",
    )
    parser.add_argument(
        "--strict-failures",
        action="store_true",
        help="Raise on agent failures instead of skipping rollouts.",
    )
    parser.add_argument(
        "--debug-agent",
        action="store_true",
        help="Log tail messages when parsing fails.",
    )
    parser.add_argument(
        "--allow-fallback-plan",
        action="store_true",
        help="Use a deterministic fallback meal plan if parsing fails.",
    )

    args = parser.parse_args()

    # Get the appropriate configuration
    config_functions = {
        "fast": config_train_fast,
        "qwen": config_train_qwen,
        "llama": config_train_llama,
        "npu": config_train_npu,
    }
    config = config_functions[args.config]()

    # Set active agent - use provided value or default based on config choice
    active_agent = args.active_agent

    print(f"Starting training with '{args.config}' configuration...")
    print(f"Active agent: {active_agent}")

    train(
        config,
        active_agent,
        args.agent_type,
        args.strict_failures,
        args.debug_agent,
        args.allow_fallback_plan,
    )


if __name__ == "__main__":
    main()