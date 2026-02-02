# High-Quality SFT Data Generation for Nutrition Agent

## Problem Statement

The goal is to create a finetuning dataset that teaches the model to output meal plans that achieve high rewards (close to 1.0).

## Key Insight from Your Example

Looking at your example meal plan (Day 0, target: 3539 kcal, 191g P, 472g C, 98g F):

| Meal | Base Cal | Quantity | Actual Cal | Actual Protein |
|------|----------|----------|------------|----------------|
| Apple Cinnamon Protein Oats | 410 | 2.0 | 820 | 52g |
| Vegetable Chickpea Curry with Rice | 500 | 2.2 | 1100 | 41.8g |
| Beef Stir-Fry with Vegetables and Rice | 580 | 1.6 | 928 | 57.6g |
| Avocado Toast with Smoked Salmon | 430 | 1.6 | 688 | 38.4g |
| **Total** | | | **~3536** | **~190g** |

**Key observations:**
1. **4 meals** per day (not 3)
2. **Individual quantities** per meal (2.0, 2.2, 1.6, 1.6) - not uniform scaling
3. **Diverse protein sources**: plant (oats), plant (chickpeas), beef, salmon
4. **Precise macro targeting**: Hits targets within ~1% error

## Solution: Mathematical Optimization

The improved `generate_sft_data.py` uses **scipy.optimize** to find optimal quantities that minimize macro error:

### Algorithm

```python
def optimize_quantities(meals, target_cal, target_prot, target_carb, target_fat):
    """
    Optimization objective: minimize weighted relative errors
    
    minimize: Σ weights[i] * ((actual[i] - target[i]) / target[i])²
    
    subject to: 0.5 ≤ quantity[i] ≤ 3.0  (realistic portion sizes)
    """
    # Build macro matrix: each row is [calories, protein, carbs, fat] for a meal
    # Solve: macro_matrix @ quantities ≈ targets
```

### Diversity Selection

Before optimization, we select diverse meals based on:
1. **Protein source**: chicken, beef, fish, plant-based, eggs, dairy
2. **Meal type**: breakfast, lunch, dinner

This ensures the model learns to create varied plans, not just repeat the same meal.

## Reward Function

The combined reward has these components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Schema | Gate | Must have valid JSON structure with required fields |
| Macro Accuracy | 0.4 | All macros within ±5% of targets |
| Variety Heuristic | 0.3 | ≥3 meals, ≥3 unique meal names |
| LLM Variety Judge | 0.3 | External LLM rates diversity (skipped in batch mode) |

**Maximum possible score**: 1.0 when all components pass.

## Usage

```bash
# Generate SFT data with default settings
cd examples/nutrition_agent/src/nutrition
python generate_sft_data.py

# With custom options
python generate_sft_data.py \
    --input ../../data/fitness_scenarios_train.parquet \
    --output ../../data/nutrition_sft_data_optimized.jsonl \
    --min-reward 0.85 \
    --limit 100 \
    --verbose
```

## Output Format

Each SFT entry is a conversation with tool calls:

```json
{
  "messages": [
    {"role": "system", "content": "You are a nutrition planner..."},
    {"role": "user", "content": "User Profile/Context:\n{\"daily_cal_target\":3539,...}\n\nRequest: Create a meal plan..."},
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "recipe_semantic_search", "arguments": "{\"meal_query\":\"muscle_gain high protein meal\",\"k\":5}"}}]},
    {"role": "tool", "name": "recipe_semantic_search", "content": "[{\"name\":\"Apple Cinnamon Protein Oats\",\"calories\":410,...}]"},
    {"role": "assistant", "content": "Based on the search results, I'll select diverse meals and optimize quantities to hit the macro targets. The plan achieves approximately 3536 kcal and 190g protein.", "tool_calls": [{"function": {"name": "return_final_answer_tool", "arguments": "{\"answer\":{\"meals\":[...]}}"}}]},
    {"role": "tool", "name": "return_final_answer_tool", "content": "{\"meals\":[{\"name\":\"Apple Cinnamon Protein Oats\",\"quantity\":2.0,\"calories\":820,...}]}"}
  ]
}
```

## Why This Works

1. **Mathematical Optimization**: Instead of guessing quantities, we solve for the optimal values
2. **Diversity Awareness**: Categorizes meals and selects different protein sources
3. **Reward Validation**: Only keeps examples that score ≥0.85 on the actual reward function
4. **Realistic Portions**: Quantities bounded between 0.5x and 3.0x servings

## Expected Results

With optimization-based generation, you should see:
- **Higher average reward**: ~0.9+ vs ~0.5-0.7 with naive generation
- **More consistent macro accuracy**: Usually within ±3% of targets
- **Better variety**: 4 diverse meals vs 3 similar meals
- **Cleaner training signal**: Model learns from high-quality examples only
