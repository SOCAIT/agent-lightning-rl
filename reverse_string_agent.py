# This agent will learn to reverse a string and also replace 'o' and 'i' letter (including uppercase) with 0 and 1 respectively.

import openai
import json
import re
from typing import List, cast

from pydantic import BaseModel, Field
from rich.console import Console

from agentlightning.adapter import TraceToMessages
from agentlightning.litagent import rollout
from agentlightning.reward import find_final_reward
from agentlightning.runner import LitAgentRunner
from agentlightning.store import InMemoryLightningStore
from agentlightning.tracer.agentops import AgentOpsTracer
from agentlightning.types import Dataset, PromptTemplate

import agentlightning as agl

console = Console()

class ReverseStringTask(BaseModel):
    input_string: str = Field(description="The input string to reverse and replace 'o' and 'i' with 0 and 1 respectively")

def parse_response_and_reward(input_string: str, content: str):
    # 1) Build the exact target output
    rev = input_string[::-1]
    trans = str.maketrans({'o': '0', 'O': '0', 'i': '1', 'I': '1'})
    target = rev.translate(trans)

    # 2) Reverse correctness (ignoring o/i case loss due to digit conversion)
    # Undo digits -> letters, then compare lowercased
    inv = str.maketrans({'0': 'o', '1': 'i'})
    content_unmapped = content.translate(inv)

    reverse_reward = 1 if content_unmapped.lower() == rev.lower() else 0

    # 3) Replacement correctness (how correct are the digits where needed)
    # Only meaningful if reversal is correct (optional, but matches your intention)
    if reverse_reward == 1:
        oi_positions = [idx for idx, ch in enumerate(rev) if ch in "oOiI"]
        if not oi_positions:
            original_string_reward = 1  # nothing to replace, treat as success
        else:
            correct = 0
            for idx in oi_positions:
                expected = '0' if rev[idx] in "oO" else '1'
                if idx < len(content) and content[idx] == expected:
                    correct += 1
            original_string_reward = correct / len(oi_positions)  # fractional reward
    else:
        original_string_reward = 0

    total_reward = 0.7 * reverse_reward + 0.3 * original_string_reward

    return total_reward, reverse_reward, original_string_reward


def build_reverse_string_agent(task, prompt):
    client = OpenAI()
    model = "gpt-5-mini"
    messages = [{'role': 'user', 'content': prompt.format(**task)}]

    client.chat.completions.create(
        model=model,
        messages=messages,
        # tools=tools,
        # tool_choice="auto",
    )

   
    final_choice = response_message.content

    total_reward, reverse_reward, original_string_reward = parse_response_and_reward(task['input_string'],  final_choice)

    return total_reward, reverse_reward, original_string_reward

def prompt_template_baseline() -> PromptTemplate:
    return PromptTemplate(
        template="Reverse the string {input_string} and replace 'o' and 'i' with 0 and 1 respectively",
        engine="f-string",
    )

import random
import string

def make_dataset(n=1000, seed=42):
    random.seed(seed)

    base_phrases = [
        "hello", "world", "hello world", "Ioannis", "Solabl", "SyntraFit",
        "offline reinforcement learning", "policy iteration", "imitation learning",
        "optimization", "information", "observation", "vision", "audio", "robotics",
        "Nicosia Cyprus", "OpenAI", "ChatGPT", "I/O bound", "input output",
        "o i O I", "oo ii OO II", "OIOIOI", "illusion", "oasis", "indigo", "ionic", "oil"
    ]

    punct = ["", "!", "!!", "???", "...", ".", ",", ";", ":", " - ", " / ", " (test)", " [ok]", " {x}"]
    casing = ["lower", "upper", "title", "mixed"]
    extras = ["", "", "", "  ", "\t", "\n", "   "]  # add occasional whitespace weirdness

    def apply_case(s, mode):
        if mode == "lower": return s.lower()
        if mode == "upper": return s.upper()
        if mode == "title": return s.title()
        # mixed
        out = []
        for ch in s:
            if ch.isalpha() and random.random() < 0.5:
                out.append(ch.upper())
            else:
                out.append(ch.lower())
        return "".join(out)

    def random_token(min_len=1, max_len=12):
        # bias toward o/i appearing
        alphabet = string.ascii_lowercase + "oi" * 3
        L = random.randint(min_len, max_len)
        return "".join(random.choice(alphabet) for _ in range(L))

    dataset = []
    for _ in range(n):
        # build a sentence from phrase + random tokens
        parts = []

        # choose 1-3 base phrases
        for __ in range(random.randint(1, 3)):
            parts.append(random.choice(base_phrases))

        # add 0-3 random tokens
        for __ in range(random.randint(0, 3)):
            parts.append(random_token())

        s = " ".join(parts)

        # apply random casing
        s = apply_case(s, random.choice(casing))

        # add punctuation and extra whitespace
        s = random.choice(extras) + s + random.choice(punct) + random.choice(extras)

        dataset.append({"input_string": s})

    return dataset

DATASET = make_dataset(n=5000, seed=123)

def load_tasks() -> Dataset:
    """Load tasks as a Dataset. Returns a list which satisfies the Dataset Protocol."""
    tasks: List[ReverseStringTask] = []

    for task in DATASET:
        tasks.append(ReverseStringTask(**task))

    # Dataset is a Protocol, so we can return the list directly
    # Lists already implement __len__ and __getitem__ which satisfy the Protocol
    return cast(Dataset[ReverseStringTask], tasks)

@agl.rollout
def reverse_string_rollout(task: ReverseStringTask, prompt_template: PromptTemplate) -> float:
    """
    Rollout function for reverse string task.
    
    Expected signature: (task, prompt_template) -> float
    Returns the total reward as a float.
    """
    from wandb_logging import log_reward_components
    
    client = OpenAI()
    model = "gpt-4o-mini"

    user_message = prompt_template.format(**task.dict())
    messages = [{'role': 'user', 'content': user_message}]

    console.print(f"[bold yellow]=== User Message ===[/bold yellow]")
    console.print(user_message)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        # tools=tools,
        # tool_choice="auto",
    )

    console.print(f"[bold yellow]=== Assistant Message ===[/bold yellow]")
    console.print(response.choices[0].message)

   
    final_choice = response.choices[0].message.content

    total_reward, reverse_reward, original_string_reward = parse_response_and_reward(
        task.input_string, final_choice
    )
    
    # Log reward components to wandb if hook is initialized
    try:
        log_reward_components(total_reward, reverse_reward, original_string_reward)
    except Exception as e:
        # If wandb logging fails, continue without logging
        console.print(f"[red]Warning: Failed to log to wandb: {e}[/red]")

    # Return only the total reward as a float (required by agentlightning)
    return total_reward