from pathlib import Path
from datasets import load_dataset
import json
from typing import Literal, Optional, List
from pydantic import BaseModel

class Scenario(BaseModel):
   question: str
   split: Literal["train", "test"]
   id: str
   daily_cal_target: Optional[int] = None
   daily_prot_target: Optional[int] = None
   daily_carb_target: Optional[int] = None
   daily_fat_target: Optional[int] = None
   banned_keywords: Optional[List[str]] = None


project_root = Path(__file__).parent.parent.parent

# Load the dataset from the JSONL file
dataset_path = project_root / "data" / "fitness_scenarios.jsonl"
if not dataset_path.exists():
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

dataset = load_dataset("json", data_files=str(dataset_path))

# Assuming the dataset has a "train" split, you can access it like this:
training_scenarios = dataset["train"]

print("Dataset loaded successfully!")
print(training_scenarios)

training_scenarios[-1]


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def one_day_meal_question(example):
    one_day_prompt = "generate a one day meal plan for user that match its macros and diet"
    example['input_question'] = f"{one_day_prompt}"
    return example

def combine_question_and_context(example):
    context_str = json.dumps(example['context'])
    example['question'] = f"{example['input_question']} Context: {context_str}"
    return example

def convert_val_to_test(example):
    if example['split'] == 'val':
       example['split'] = 'test'
    return example

def get_target_nutrition_data(example):
    daily_cal_target = example['context']['daily_cal_target']
    daily_prot_target = example['context']['daily_prot_target']
    daily_carb_target = example['context']['daily_carb_target']
    daily_fat_target = example['context']['daily_fat_target']
    banned_keywords = example['context']['banned_keywords']

    example['daily_cal_target'] = daily_cal_target
    example['daily_prot_target'] = daily_prot_target
    example['daily_carb_target'] = daily_carb_target
    example['daily_fat_target'] = daily_fat_target
    example['banned_keywords'] = banned_keywords
    return example

def extract_context_columns(example):
    context = example['context']
    example['age'] = context.get('age')
    example['sex'] = context.get('sex')
    example['height_cm'] = context.get('height_cm')
    example['weight_kg'] = context.get('weight_kg')
    example['goal'] = context.get('goal')
    example['activity'] = context.get('activity')
    example['dietary_prefs'] = context.get('dietary_prefs')
    example['equipment'] = context.get('equipment')
    example['experience'] = context.get('experience')
    return example

def make_scenario(example):
    scenario = Scenario(
        question=example['question'],
        split=example['split'],
        id=str(example['id']),
        daily_cal_target=example['daily_cal_target'],
        daily_prot_target=example['daily_prot_target'],
        daily_carb_target=example['daily_carb_target'],
        daily_fat_target=example['daily_fat_target'],
        #banned_keywords=example['banned_keywords']
    )
    return scenario
training_scenarios = training_scenarios.map(one_day_meal_question)
training_scenarios = training_scenarios.map(combine_question_and_context)
training_scenarios = training_scenarios.map(convert_val_to_test)
training_scenarios = training_scenarios.map(get_target_nutrition_data)
training_scenarios = training_scenarios.map(extract_context_columns)
# training_scenarios = training_scenarios.map(make_scenario)
print(training_scenarios[0]['question'])

scenarios_list = []
for example in training_scenarios:
    scenario = Scenario(
        question=example['question'],
        split=example['split'],
        id=str(example['id']),
        daily_cal_target=example['daily_cal_target'],
        daily_prot_target=example['daily_prot_target'],
        daily_carb_target=example['daily_carb_target'],
        daily_fat_target=example['daily_fat_target'],
        banned_keywords=example['banned_keywords']
    )
    scenarios_list.append(scenario)

print(f"Created a list of {len(scenarios_list)} Scenario objects.")

from collections import Counter

print(Counter(training_scenarios['split']))

scenarios_list[0]