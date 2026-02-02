import os
import argparse
from typing import Dict, Any, List, Optional
import json
import logging

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_chat_template(example: Dict[str, Any], tokenizer) -> str:
    """
    Formats the conversation using the tokenizer's chat template.
    Expects example["messages"] to be a list of dicts.
    """
    messages = example["messages"]
    # We want to train on the assistant's responses and tool calls.
    # DataCollatorForCompletionOnlyLM will handle masking user/system prompts if we set it up right.
    # But for simplicity with simple SFT, we often train on the whole sequence or just use the collator.
    
    # Qwen-2.5 and Llama-3 have specific chat templates.
    # We rely on tokenizer.apply_chat_template if available.
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return text
    except Exception as e:
        logger.warning(f"Failed to apply chat template: {e}. Fallback to manual.")
        # Fallback manual formatting (ChatML style)
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            
            if role == "system":
                text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                text += "<|im_start|>assistant\n"
                if tool_calls:
                    # Format tool calls
                    for tc in tool_calls:
                        func = tc["function"]
                        text += f"<tool_code>{func['name']}({func['arguments']})</tool_code>\n"
                else:
                    text += f"{content}\n"
                text += "<|im_end|>\n"
            elif role == "tool":
                text += f"<|im_start|>tool\n{content}<|im_end|>\n"
        return text

def train_sft(
    model_name: str,
    data_path: str,
    output_dir: str,
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_seq_length: int = 2048,
):
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Check chat template support
    if not tokenizer.chat_template:
        logger.warning("Tokenizer has no chat template. Using default ChatML.")
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"

    logger.info(f"Loading dataset from {data_path}")
    # Load JSONL
    data = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    dataset = Dataset.from_list(data)
    
    # Format dataset
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["messages"])):
            # HuggingFace datasets passes batches, so we need to handle the list structure if batched
            # But usually map passes single examples if batched=False
            pass 
        return [format_chat_template({"messages": msgs}, tokenizer) for msgs in example["messages"]]

    # Use DataCollatorForCompletionOnlyLM to train only on Assistant outputs?
    # For tool use, we want to train on the tool call generation AND the final answer generation.
    # The simple way is standard SFT on the whole conversation, or masking user/system.
    # "response_template" for Qwen might be "<|im_start|>assistant\n"
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=True, # or bf16 if supported
        bf16=torch.cuda.is_bf16_supported(),
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model_name,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data", type=str, required=True, help="Path to sft jsonl file")
    parser.add_argument("--output", type=str, default="nutrition_sft_model")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    train_sft(args.model, args.data, args.output, args.epochs)
