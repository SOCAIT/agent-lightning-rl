# Simple Wandb Logging Guide

If the full `WandbLoggingHook` is too complex or causing issues, use this simpler approach.

## Quick Start

### 1. Use `simple_wandb.py` (Recommended for simplicity)

```python
from simple_wandb import init_wandb, log_reward, finish

# At start of training
init_wandb(project="my-project", config={"lr": 0.001})

# In your rollout function - just log the reward
from simple_wandb import log_reward

@agl.rollout
def my_rollout(task, prompt_template):
    # ... your code ...
    reward = calculate_reward(...)
    log_reward(reward)  # That's it!
    return reward

# At end of training
finish()
```

### 2. Full Example

See `reverse_string_apo_simple.py` for a complete example.

## What's Fixed

1. **Task handling**: Now handles both dict and object formats
2. **Async hooks**: All hook methods are now async (required by agentlightning)
3. **Simple logging**: `simple_wandb.py` avoids complex hooks entirely

## Two Options

### Option A: Simple (No Hooks)
- Use `simple_wandb.py`
- Just call `init_wandb()` at start
- Call `log_reward()` from your rollout function
- No hooks needed in Trainer

### Option B: Full Hook System
- Use `wandb_logging.py` with `WandbLoggingHook`
- More features (epoch tracking, batch stats, etc.)
- Requires hooks in Trainer

## Which Should You Use?

- **Use `simple_wandb.py`** if:
  - You just want to log rewards
  - You're having issues with hooks
  - You want minimal code

- **Use `wandb_logging.py`** if:
  - You need detailed metrics (epochs, batches, etc.)
  - You want automatic tracking
  - You're comfortable with hooks

## Current Status

The rollout function (`reverse_string_rollout`) now:
1. Handles both dict and object task formats ✅
2. Tries simple_wandb first, falls back to wandb_logging ✅
3. Silently continues if wandb fails ✅

The hook (`WandbLoggingHook`) now:
1. Has async hook methods ✅
2. Handles missing methods gracefully ✅
3. Logs rewards from rollout_end when available ✅

