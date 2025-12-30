from reverse_string_agent import load_tasks, prompt_template_baseline, build_reverse_string_agent
from reverse_string_agent import ReverseStringTask

import logging
from typing import Tuple, cast, Any, Dict, Optional

from openai import AsyncOpenAI

from agentlightning import Trainer, setup_logging
from agentlightning.adapter import TraceToMessages
from agentlightning.algorithm.apo import APO
from agentlightning.types import Dataset

from wandb_logging import WandbLoggingHook, set_global_hook


def setup_apo_logger(file_path: str = "apo.log") -> None:
    """Dump a copy of all the logs produced by APO algorithm to a file."""

    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] (Process-%(process)d %(name)s)   %(message)s")
    file_handler.setFormatter(formatter)
    logging.getLogger("agentlightning.algorithm.apo").addHandler(file_handler)

def load_train_val_data() -> Tuple[Dataset, Dataset]:
    dataset = load_tasks()
    train_val_Split = len(dataset) * 0.8 
    train_data = dataset[:train_val_Split]
    val_data = dataset[train_val_Split:]
    return train_data, val_data

def main() -> None:
    setup_apo_logger()
    train_data, val_data = load_train_val_data()
    openai_client = AsyncOpenAI()

    # Configure APO algorithm hyperparameters
    algo_config = {
        "val_batch_size": 16,
        "gradient_batch_size": 4,
        "beam_width": 2,
        "branch_factor": 2,
        "beam_rounds": 2,
    }

    algo = APO[ReverseStringTask](
        openai_client=openai_client,
        val_batch_size=algo_config["val_batch_size"],
        gradient_batch_size=algo_config["gradient_batch_size"],
        beam_width=algo_config["beam_width"],
        branch_factor=algo_config["branch_factor"],
        beam_rounds=algo_config["beam_rounds"],
        _poml_trace=True,
    )

    # Initialize wandb logging hook with config
    monitoring_hook = WandbLoggingHook(
        project_name="reverse-string-apo",
        entity=None,  # Set to your wandb username/team if needed
        run_name="reverse-string-apo-run",
        config={
            **algo_config,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "n_runners": 8,
        }
    )
    
    # Set as global hook for easy access from rollout functions
    set_global_hook(monitoring_hook)

    trainer = Trainer(
        algorithm=algo,
        n_runners=4,
        strategy="cs",
        adapter=TraceToMessages(),
        hooks=[monitoring_hook]
    )

    # Start training
    try:
        trainer.fit(
            algorithm=algo,
            # increase number of runners to run more rollouts in parallel
            n_runners=8,
            # APO algorithm needs a baseline
            # set it either here or in the algo,
            initial_resources={
                "prompt": prompt_template_baseline(),
            },
            # APO algorithm needs an adapter to process the traces produced by the rollout
            adapter=TraceToMessages(),
        )
    finally:
        # Ensure wandb finishes properly
        if monitoring_hook.initialized:
            monitoring_hook.on_train_end(trainer)


if __name__ == "__main__":
    main()