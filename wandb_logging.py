"""
Wandb logging utilities for agentlightning training.

This module provides hooks and utilities for logging training metrics,
rewards, and other information to Weights & Biases.
"""

import logging
from typing import Any, Dict, Optional, List
import wandb
from collections import defaultdict


class WandbLoggingHook:
    """Custom hook for logging training metrics to Weights & Biases."""
    
    def __init__(
        self,
        project_name: str = "reverse-string-apo",
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize wandb logging hook.
        
        Args:
            project_name: W&B project name
            entity: W&B entity (username or team)
            run_name: Name for this specific run
            config: Dictionary of hyperparameters/config to log
        """
        self.project_name = project_name
        self.entity = entity
        self.run_name = run_name
        self.config = config or {}
        self.step = 0
        self.initialized = False
        self.reward_history: List[float] = []
        self.batch_rewards: List[float] = []
        
    def on_train_start(self, trainer: Any) -> None:
        """Called when training starts."""
        if not self.initialized:
            wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=self.run_name,
                config=self.config,
            )
            self.initialized = True
            logging.info(f"Wandb initialized: project={self.project_name}, run={self.run_name}")
    
    def on_train_end(self, trainer: Any) -> None:
        """Called when training ends."""
        if self.initialized:
            # Log final summary statistics
            if self.reward_history:
                wandb.summary["final_avg_reward"] = sum(self.reward_history) / len(self.reward_history)
                wandb.summary["max_reward"] = max(self.reward_history)
                wandb.summary["min_reward"] = min(self.reward_history)
            
            wandb.finish()
            logging.info("Wandb run finished")
    
    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Called at the start of each epoch."""
        self.current_epoch = epoch
        self.epoch_rewards: List[float] = []
        wandb.log({"epoch": epoch}, step=self.step)
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Optional[Dict[str, float]] = None) -> None:
        """Called at the end of each epoch."""
        # Log epoch-level statistics
        if self.epoch_rewards:
            epoch_metrics = {
                "epoch": epoch,
                "reward/epoch_mean": sum(self.epoch_rewards) / len(self.epoch_rewards),
                "reward/epoch_max": max(self.epoch_rewards),
                "reward/epoch_min": min(self.epoch_rewards),
            }
            if metrics:
                epoch_metrics.update(metrics)
            wandb.log(epoch_metrics, step=self.step)
        
        self.step += 1
    
    def on_batch_end(
        self,
        trainer: Any,
        batch_idx: int,
        reward: Optional[float] = None,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Called at the end of each batch."""
        log_dict = {}
        
        if reward is not None:
            log_dict["reward"] = reward
            log_dict["reward/batch"] = reward
            self.reward_history.append(reward)
            self.batch_rewards.append(reward)
            if hasattr(self, 'epoch_rewards'):
                self.epoch_rewards.append(reward)
        
        if loss is not None:
            log_dict["loss"] = loss
            log_dict["loss/batch"] = loss
        
        if metrics:
            log_dict.update(metrics)
        
        if log_dict:
            log_dict["batch"] = batch_idx
            # Log moving averages
            if len(self.batch_rewards) > 0:
                window_size = min(10, len(self.batch_rewards))
                recent_rewards = self.batch_rewards[-window_size:]
                log_dict["reward/moving_avg"] = sum(recent_rewards) / len(recent_rewards)
            
            wandb.log(log_dict, step=self.step)
            self.step += 1
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log custom metrics."""
        if step is None:
            step = self.step
        wandb.log(metrics, step=step)
    
    def log_reward(
        self,
        reward: float,
        reward_type: str = "total",
        step: Optional[int] = None,
        additional_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log reward value with optional additional metrics.
        
        Args:
            reward: The reward value to log
            reward_type: Type of reward (e.g., "total", "reverse", "replacement")
            step: Optional step number (uses internal step counter if None)
            additional_metrics: Optional dict of additional metrics to log
        """
        if step is None:
            step = self.step
        
        log_dict = {f"reward/{reward_type}": reward}
        
        # Track reward in history
        self.reward_history.append(reward)
        if hasattr(self, 'epoch_rewards'):
            self.epoch_rewards.append(reward)
        
        if additional_metrics:
            log_dict.update(additional_metrics)
        
        wandb.log(log_dict, step=step)
        self.step += 1
    
    def log_validation_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log validation metrics."""
        if step is None:
            step = self.step
        # Prefix validation metrics with "val/"
        val_metrics = {f"val/{k}": v for k, v in metrics.items()}
        wandb.log(val_metrics, step=step)
    
    def log_reward_components(
        self,
        total_reward: float,
        reverse_reward: float,
        replacement_reward: float,
        step: Optional[int] = None,
    ) -> None:
        """
        Log reward components separately.
        
        Useful when you have multiple reward components (e.g., reverse_reward, replacement_reward).
        """
        if step is None:
            step = self.step
        
        wandb.log({
            "reward/total": total_reward,
            "reward/reverse": reverse_reward,
            "reward/replacement": replacement_reward,
        }, step=step)
        
        self.reward_history.append(total_reward)
        if hasattr(self, 'epoch_rewards'):
            self.epoch_rewards.append(total_reward)
        
        self.step += 1
    
    # Agentlightning hook methods - these are called by the Trainer
    def on_trace_start(self, *args, **kwargs) -> None:
        """Called when a trace starts. Optional hook method."""
        pass
    
    def on_trace_end(self, *args, **kwargs) -> None:
        """Called when a trace ends. Optional hook method."""
        pass
    
    def on_rollout_start(self, *args, **kwargs) -> None:
        """Called when a rollout starts. Optional hook method."""
        pass
    
    def on_rollout_end(self, *args, **kwargs) -> None:
        """Called when a rollout ends. Optional hook method."""
        pass
    
    def on_span_start(self, *args, **kwargs) -> None:
        """Called when a span starts. Optional hook method."""
        pass
    
    def on_span_end(self, *args, **kwargs) -> None:
        """Called when a span ends. Optional hook method."""
        pass
    
    def __getattr__(self, name: str) -> Any:
        """
        Handle any other hook methods that agentlightning might call.
        Returns a no-op function for any undefined hook methods.
        """
        if name.startswith('on_'):
            # Return a no-op function for any 'on_*' hook methods we haven't defined
            def noop(*args, **kwargs):
                pass
            return noop
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


# Global hook instance for easy access
_global_hook: Optional[WandbLoggingHook] = None


def set_global_hook(hook: WandbLoggingHook) -> None:
    """Set a global hook instance for easy access from anywhere."""
    global _global_hook
    _global_hook = hook


def get_global_hook() -> Optional[WandbLoggingHook]:
    """Get the global hook instance."""
    return _global_hook


def log_reward(reward: float, reward_type: str = "total", **kwargs) -> None:
    """
    Convenience function to log rewards using the global hook.
    
    Usage:
        from wandb_logging import log_reward
        log_reward(0.85, reward_type="total")
    """
    hook = get_global_hook()
    if hook and hook.initialized:
        hook.log_reward(reward, reward_type=reward_type, **kwargs)
    else:
        logging.warning("Wandb hook not initialized. Cannot log reward.")


def log_reward_components(total_reward: float, reverse_reward: float, replacement_reward: float) -> None:
    """
    Convenience function to log reward components using the global hook.
    
    Usage:
        from wandb_logging import log_reward_components
        log_reward_components(0.85, 1.0, 0.5)
    """
    hook = get_global_hook()
    if hook and hook.initialized:
        hook.log_reward_components(total_reward, reverse_reward, replacement_reward)
    else:
        logging.warning("Wandb hook not initialized. Cannot log reward components.")

