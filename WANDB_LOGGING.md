# Wandb Logging Guide

This guide explains how to log rewards and other metrics during training using Weights & Biases (wandb).

## Setup

1. **Install wandb**:
   ```bash
   pip install wandb
   ```

2. **Login to wandb** (first time only):
   ```bash
   wandb login
   ```
   You'll need to get your API key from [wandb.ai/signup](https://wandb.ai/signup)

## Basic Usage

### 1. Using the WandbLoggingHook

The `WandbLoggingHook` is automatically integrated into your training script. It will log:
- Rewards (total, reverse, replacement components)
- Loss values
- Epoch and batch statistics
- Custom metrics

```python
from wandb_logging import WandbLoggingHook, set_global_hook

# Initialize the hook
monitoring_hook = WandbLoggingHook(
    project_name="reverse-string-apo",
    entity="your-username",  # Optional: your wandb username/team
    run_name="experiment-1",
    config={
        "val_batch_size": 16,
        "gradient_batch_size": 4,
        # ... other hyperparameters
    }
)

# Set as global hook for easy access
set_global_hook(monitoring_hook)

# Add to trainer
trainer = Trainer(
    algorithm=algo,
    hooks=[monitoring_hook],
    # ... other parameters
)
```

### 2. Logging Rewards from Rollout Functions

You can log rewards directly from your rollout functions using convenience functions:

```python
from wandb_logging import log_reward, log_reward_components
from agentlightning.litagent import rollout

@rollout
def my_rollout(task, prompt):
    # ... your rollout code ...
    
    # Calculate rewards
    total_reward, reverse_reward, replacement_reward = calculate_rewards(...)
    
    # Log all reward components
    log_reward_components(total_reward, reverse_reward, replacement_reward)
    
    # Or log individual rewards
    log_reward(total_reward, reward_type="total")
    
    return total_reward
```

### 3. Manual Logging

For more control, you can use the hook directly:

```python
from wandb_logging import get_global_hook

hook = get_global_hook()

# Log custom metrics
hook.log_metrics({
    "custom_metric": 0.95,
    "another_metric": 42.0,
})

# Log reward with additional context
hook.log_reward(
    reward=0.85,
    reward_type="total",
    additional_metrics={
        "episode_length": 10,
        "success_rate": 0.8,
    }
)

# Log validation metrics
hook.log_validation_metrics({
    "val_accuracy": 0.92,
    "val_loss": 0.15,
})
```

## What Gets Logged

The hook automatically logs:

- **Rewards**:
  - `reward/total` - Total reward
  - `reward/reverse` - Reverse correctness reward
  - `reward/replacement` - Replacement correctness reward
  - `reward/batch` - Batch-level rewards
  - `reward/epoch_mean` - Average reward per epoch
  - `reward/moving_avg` - Moving average of recent rewards

- **Training Metrics**:
  - `loss` - Training loss
  - `loss/batch` - Batch-level loss
  - `epoch` - Current epoch number
  - `batch` - Current batch number

- **Validation Metrics** (prefixed with `val/`):
  - `val/accuracy`
  - `val/loss`
  - Any custom validation metrics

- **Summary Statistics** (logged at end of training):
  - `final_avg_reward` - Average reward across all training
  - `max_reward` - Maximum reward achieved
  - `min_reward` - Minimum reward achieved

## Viewing Results

1. **Web Interface**: Go to [wandb.ai](https://wandb.ai) and navigate to your project
2. **Command Line**: The wandb run URL will be printed when training starts
3. **Local Dashboard**: Run `wandb local` to start a local dashboard

## Examples

See `example_wandb_logging.py` for complete examples of:
- Logging from rollout functions
- Using the hook directly
- Logging during custom training loops

## Troubleshooting

### Hook not initialized
If you see "Wandb hook not initialized" warnings:
- Make sure you call `set_global_hook(monitoring_hook)` before using convenience functions
- Ensure `monitoring_hook.on_train_start()` is called (happens automatically when training starts)

### Metrics not appearing
- Check that wandb is properly initialized: `wandb.run` should not be None
- Verify you're calling `wandb.log()` or using the hook's logging methods
- Check the wandb dashboard for any errors

### Authentication issues
- Run `wandb login` again if you get authentication errors
- Make sure your API key is valid

## Advanced Usage

### Custom Hook Methods

The hook provides several methods you can override or use:

- `on_train_start(trainer)` - Called when training starts
- `on_train_end(trainer)` - Called when training ends
- `on_epoch_start(trainer, epoch)` - Called at start of each epoch
- `on_epoch_end(trainer, epoch, metrics)` - Called at end of each epoch
- `on_batch_end(trainer, batch_idx, reward, loss, metrics)` - Called at end of each batch
- `log_metrics(metrics, step)` - Log custom metrics
- `log_reward(reward, reward_type, step, additional_metrics)` - Log reward
- `log_validation_metrics(metrics, step)` - Log validation metrics

### Accessing Reward History

The hook maintains reward history:

```python
hook = get_global_hook()
print(f"Total rewards logged: {len(hook.reward_history)}")
print(f"Average reward: {sum(hook.reward_history) / len(hook.reward_history)}")
```

