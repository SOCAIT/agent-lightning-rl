"""
Simple, minimalistic wandb logging for agentlightning.

This is a simplified version that just logs rewards without complex hooks.
Usage:
    from simple_wandb import init_wandb, log_reward
    
    # At start of training
    init_wandb(project="my-project", config={"lr": 0.001})
    
    # In your rollout function
    log_reward(0.85)
"""

import wandb
from typing import Optional, Dict, Any

_wandb_initialized = False


def init_wandb(
    project: str = "agentlightning-training",
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    entity: Optional[str] = None,
) -> None:
    """Initialize wandb. Call this once at the start of training."""
    global _wandb_initialized
    if not _wandb_initialized:
        wandb.init(
            project=project,
            name=name,
            config=config or {},
            entity=entity,
        )
        _wandb_initialized = True


def log_reward(reward: float, step: Optional[int] = None) -> None:
    """Log a reward value. Simple and straightforward."""
    if not _wandb_initialized:
        return  # Silently skip if not initialized
    
    log_dict = {"reward": reward}
    if step is not None:
        wandb.log(log_dict, step=step)
    else:
        wandb.log(log_dict)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log any metrics dictionary."""
    if not _wandb_initialized:
        return
    
    if step is not None:
        wandb.log(metrics, step=step)
    else:
        wandb.log(metrics)


def finish() -> None:
    """Finish wandb run. Call at end of training."""
    global _wandb_initialized
    if _wandb_initialized:
        wandb.finish()
        _wandb_initialized = False

