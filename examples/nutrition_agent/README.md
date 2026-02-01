# Nutrition Agent Example

This example demonstrates how to train a nutrition agent using `agentlightning`.

## Setup

1.  **Navigate to this directory:**

    ```bash
    cd examples/nutrition_agent
    ```

2.  **Install dependencies:**

    It is recommended to use a virtual environment or ensure you don't conflict with the root project dependencies if they differ. However, since this is an example within the repo, you might simply want to install the extra requirements.

    ```bash
    pip install -r requirements.txt
    ```

    Ensure `agentlightning` is installed (e.g., via `pip install -e ../..` from this directory if you are developing, or just `pip install agentlightning` if using the published package).

3.  **Environment Variables:**

    Create a `.env` file in this directory if needed, for keys like `OPENAI_API_KEY` or `PINECONE_API_KEY`.

## Running the Training

Run the training script from this directory:

```bash
python src/nutrition/train_nutrition_agent.py qwen
```

### Configuration Options:

- `qwen`: **Recommended** - Configuration for Qwen-2.5-7B-Instruct with optimized sequence lengths:
  - Model: `Qwen/Qwen2.5-7B-Instruct`
  - `max_prompt_length`: 12288 tokens (supports multi-turn tool calls)
  - `max_response_length`: 4096 tokens (ensures complete plan generation)
- `fast`: Fast training run for testing (uses 1.5B model, 1 epoch, minimal steps).
- `llama`: Configuration for LLaMA-3.2-1B.
- `npu`: Configuration for NPU training.

### Optional Arguments:

- `--agent-type`: Choose agent implementation
  - `llm` (default) - Standard LLM agent with tool calling
  - `deterministic` - Deterministic agent
  - `deterministic-llm` - Deterministic agent with LLM optimization

- `--active-agent`: Override the active agent name (default: auto-generated)

### Example Commands:

```bash
# Standard training with Qwen 7B (recommended)
python src/nutrition/train_nutrition_agent.py qwen

# Training with deterministic agent
python src/nutrition/train_nutrition_agent.py qwen --agent-type deterministic

# Fast test run
python src/nutrition/train_nutrition_agent.py fast
```

## Directory Structure

- `src/`: Contains the agent and environment code.
- `data/`: Contains the training and validation data.
- `requirements.txt`: Dependencies for this example.
