from reverse_string_agent import load_tasks, prompt_template_baseline, build_reverse_string_agent



import logging
from typing import Tuple, cast

from openai import AsyncOpenAI

from agentlightning import Trainer, setup_logging
from agentlightning.adapter import TraceToMessages
from agentlightning.algorithm.apo import APO
from agentlightning.types import Dataset


logging.basicConfig(level=logging.INFO)
setup_logging()

def load_train_val_data() -> Tuple[Dataset, Dataset]:
    dataset = load_tasks()
    train_val_Split = len(dataset) * 0.8 
    train_data = dataset[:train_val_Split]
    val_data = dataset[train_val_Split:]
    return train_data, val_data
