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
python src/nutrition/train_nutrition_agent.py fast
```

Arguments:
- `fast`: Fast training run for testing.
- `qwen`: Configuration for Qwen-2.5B.
- `llama`: Configuration for LLaMA-3.2-1B.
- `npu`: Configuration for NPU training.

## Directory Structure

- `src/`: Contains the agent and environment code.
- `data/`: Contains the training and validation data.
- `requirements.txt`: Dependencies for this example.
