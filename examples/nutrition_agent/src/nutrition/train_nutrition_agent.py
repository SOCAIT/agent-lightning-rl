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
# B200 180GB x2
RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": True,
    },
    "data": {
        "train_files": "data/fitness_scenarios_train.parquet",
        "val_files": "data/fitness_scenarios_val.parquet",
        "train_batch_size": 16,
        "max_prompt_length": 2048,
        "max_response_length": 2048,
        "truncation": "left",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 8,
            "log_prob_micro_batch_size_per_gpu": 2,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            "gpu_memory_utilization": 0.40,
            "max_model_len": 8192,
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                    "max_num_seqs": 8,
                    "max_num_batched_tokens": 8192,
                    "enable_chunked_prefill": False, # SET TO FALSE: Fixes shape mismatches
                    "enforce_eager": True, # SET TO TRUE: Fixes vLLM/FSDP graph shape errors
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 16,
            "ppo_micro_batch_size_per_gpu": 2,
            "optim": {"lr": 1e-6},
            "use_kl_loss": True,
            "kl_loss_coef": 0.05,
            "entropy_coeff": 0.01,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {
                "param_offload": False,
                "optimizer_offload": False,
            },
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 2,
            "fsdp_config": {"param_offload": False},
        },
        "model": {
            "path": "Qwen/Qwen2.5-14B-Instruct",
            "use_remove_padding": False, # SET TO FALSE: Standardizes shapes for FSDP
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 2,
        "val_before_train": False,
        "critic_warmup": 0,
        "logger": ["console", "wandb"],
        "project_name": "AgentLightning",
        "experiment_name": "nutrition_14b_stable_grpo",
        "nnodes": 1,
        # --- ROLLING CHECKPOINT LOGIC ---
        "save_freq": 16,  
        "test_freq": 16,                  # Saves every 16 iterations
        "remove_previous_ckpt_in_save": True, # DELETES the folder from step 16 when step 32 is saved
        # "default_local_dir": "./checkpoints", # Path where the latest folder will live
        # --------------------------------
        
        "resume_mode": "auto",              # If it crashes, it starts back from the last save
        "log_val_generations": 5,           # Logs 5 agent examples to WandB every test_freq
        "total_epochs": 5,
    },
}

RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
        "kl_ctrl": {
            "kl_coef": 0.001,
        },
    },
    "data": {
        "train_files": "data/fitness_scenarios_train.parquet",
        "val_files": "data/fitness_scenarios_val.parquet",
        "train_batch_size": 116,
        "val_batch_size": 116,
        "max_prompt_length": 1024,
        "max_response_length": 2048,
        "filter_overlong_prompts": True,
        "truncation": "error",
        "shuffle": False,
    },
    "actor_rollout_ref": {
        "model": {
            "path": "Qwen/Qwen2.5-14B-Instruct",
            "use_shm": True,
            "enable_gradient_checkpointing": True,
            "use_remove_padding": True,
            "lora_rank": 32,
            "lora_alpha": 32,
            "target_modules": "all-linear",
        },
        "rollout": {
            "name": "vllm",
            "tensor_model_parallel_size": 1,
            "n": 4,
            "log_prob_micro_batch_size": 116,  # Matches bash: ${mini_batch_size}
            "gpu_memory_utilization": 0.6,
            "max_num_seqs": 512,  # From bash script
            "max_model_len": 1536,  # From bash script
            "max_num_batched_tokens": 1536,  # From bash script
            "enable_chunked_prefill": False,
            "load_format": "safetensors",
            "layered_summon": True,
            "multi_turn": {"format": "hermes"},
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 116,  # ${mini_batch_size}
            "ppo_micro_batch_size_per_gpu": 116,  # Bash uses ppo_micro_batch_size, not per_gpu
            "ulysses_sequence_parallel_size": 2,
            "optim": {"lr": 3e-5},
            "use_kl_loss": True,
            "kl_loss_coef": 0.001,
            "kl_loss_type": "low_var_kl",
            "entropy_coeff": 0.001,
            "fsdp_config": {
                "fsdp_size": -1,
                "param_offload": True,
                "optimizer_offload": True,
            },
        },
        "ref": {
            "log_prob_micro_batch_size": 116,  # ${mini_batch_size}
            "fsdp_config": {
                "param_offload": True,
            },
        },
    },
    "trainer": {
        "n_gpus_per_node": 2,
        "nnodes": 1,
        "val_before_train": False,
        "critic_warmup": 0,
        "logger": ["console", "wandb"],
        "project_name": "AgentLightning",
        "experiment_name": "nutrition_14b_multiturn_lora",
        "save_freq": 20,
        "test_freq": 10,
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
    # Fast config uses same 14B model but fewer steps
    config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.5
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