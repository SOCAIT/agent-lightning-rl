"""
Example: How to log rewards and metrics during training with wandb

This file demonstrates different ways to log rewards and metrics during training.
"""

from wandb_logging import WandbLoggingHook, set_global_hook, log_reward, log_reward_components
from reverse_string_agent import ReverseStringTask, parse_response_and_reward
from agentlightning.litagent import rollout
from agentlightning.types import PromptTemplate
from openai import OpenAI


# Example 1: Logging rewards from a rollout function
@rollout
def reverse_string_rollout_with_logging(task: ReverseStringTask, prompt: PromptTemplate) -> float:
    """
    Example rollout function that logs rewards to wandb.
    
    You can use the convenience functions log_reward() or log_reward_components()
    from anywhere in your code once the hook is initialized.
    """
    client = OpenAI()
    model = "gpt-4o-mini"
    messages = [{'role': 'user', 'content': prompt.format(**task.dict())}]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    
    final_choice = response.choices[0].message.content
    
    # Calculate rewards
    total_reward, reverse_reward, replacement_reward = parse_response_and_reward(
        task.input_string, final_choice
    )
    
    # Log all reward components to wandb
    log_reward_components(total_reward, reverse_reward, replacement_reward)
    
    # Or log individual rewards
    # log_reward(total_reward, reward_type="total")
    # log_reward(reverse_reward, reward_type="reverse")
    # log_reward(replacement_reward, reward_type="replacement")
    
    return total_reward


# Example 2: Using the hook directly for more control
def example_direct_hook_usage():
    """Example of using the hook directly for more advanced logging."""
    
    hook = WandbLoggingHook(
        project_name="my-project",
        run_name="experiment-1",
        config={
            "learning_rate": 0.001,
            "batch_size": 32,
        }
    )
    
    # Initialize wandb
    hook.on_train_start(None)
    
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
        "accuracy": 0.92,
        "loss": 0.15,
    })
    
    # Finish
    hook.on_train_end(None)


# Example 3: Logging during training loop (if you have access to training loop)
def example_training_loop_logging():
    """Example of logging during a custom training loop."""
    
    hook = WandbLoggingHook(project_name="my-project")
    hook.on_train_start(None)
    set_global_hook(hook)
    
    # Simulated training loop
    for epoch in range(10):
        hook.on_epoch_start(None, epoch)
        
        epoch_rewards = []
        for batch_idx in range(100):
            # Simulate getting a reward from your training
            reward = 0.5 + (epoch * 0.05)  # Simulated reward improving over time
            
            # Log batch-level metrics
            hook.on_batch_end(
                trainer=None,
                batch_idx=batch_idx,
                reward=reward,
                loss=1.0 - reward,  # Simulated loss
                metrics={
                    "learning_rate": 0.001 * (0.9 ** epoch),
                }
            )
            
            epoch_rewards.append(reward)
        
        # Log epoch-level summary
        hook.on_epoch_end(
            trainer=None,
            epoch=epoch,
            metrics={
                "reward/epoch_mean": sum(epoch_rewards) / len(epoch_rewards),
                "reward/epoch_max": max(epoch_rewards),
            }
        )
    
    hook.on_train_end(None)


if __name__ == "__main__":
    print("This file contains examples of wandb logging.")
    print("See the functions above for different usage patterns.")
    
    # Uncomment to run examples:
    # example_direct_hook_usage()
    # example_training_loop_logging()

