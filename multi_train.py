import os
import random
from dataclasses import dataclass
from typing import Optional, List

import torch
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

from dataset import SFTDataCollator, SFTDataset
from utils.constants import model2template

import yaml
from loguru import logger

@dataclass
class LoraTrainingArguments:
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float


def train_lora(model_id: str, context_length: int, training_args: LoraTrainingArguments):
    """
    Fine-tune a 4-bit quantized model using LoRA + Accelerate + SFTTrainer.
    
    Args:
        model_id: HuggingFace model ID
        context_length: Maximum sequence length
        training_args: Training configuration
    """
    # (1) Initialize Accelerator with optimal settings
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    # Only log on main process
    if accelerator.is_main_process:
        logger.info(f"Using device: {accelerator.device}")
        logger.info(f"Mixed precision: {accelerator.mixed_precision}")

    # (2) Define LoRA config with optimal target modules
    lora_config = LoraConfig(
        r=training_args.lora_rank,
        target_modules=[
            "qkv_proj",
            "o_proj",
            "gate_up_proj",
        ],
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        task_type="CAUSAL_LM",
    )

    # (3) 4-bit quantization config with optimal settings
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # (4) Define SFT training config
    sft_config = SFTConfig(
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_steps=100,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        report_to=[],
        run_name=f"{model_id}-lora",
        output_dir="outputs_multi",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        num_train_epochs=training_args.num_train_epochs,
        max_seq_length=context_length,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # (5) Load tokenizer with padding token
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        padding_side="right",
        token=os.environ.get("HF_TOKEN"),
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # (6) Load and prepare model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    
    if sft_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # (7) Load and prepare datasets
    train_dataset = SFTDataset(
        file="data/demo_data.jsonl",
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
    )
    eval_dataset = SFTDataset(
        file="data/demo_data.jsonl",
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
    )

    # Combine with additional data
    additional_dataset = SFTDataset(
        file="data/demo_data.jsonl",
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
    )
    additional_data_size = int(len(additional_dataset.data_list) * 1.0)
    additional_data_subset = random.sample(additional_dataset.data_list, additional_data_size)
    combined_data_list = train_dataset.data_list + additional_data_subset
    combined_dataset = SFTDataset(
        file="data/demo_data.jsonl",
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
        custom_data_list=combined_data_list,
    )

    # (8) Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=combined_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        peft_config=lora_config,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=context_length),
    )

    # Prepare everything with accelerator
    trainer = accelerator.prepare(trainer)

    # (9) Training with proper error handling
    try:
        if accelerator.is_main_process:
            logger.info("Starting training...")
        train_result = trainer.train()
        if accelerator.is_main_process:
            logger.info(f"Training completed. Metrics: {train_result.metrics}")
            # Save the final model
            trainer.save_model("outputs_multi/final")
            
    except Exception as e:
        if accelerator.is_main_process:
            logger.error(f"Training failed with error: {str(e)}")
        raise

    # (10) Evaluation
    try:
        if accelerator.is_main_process:
            logger.info("Starting evaluation...")
        eval_result = trainer.evaluate()
        if accelerator.is_main_process:
            logger.info(f"Evaluation metrics: {eval_result}")
    except Exception as e:
        if accelerator.is_main_process:
            logger.error(f"Evaluation failed with error: {str(e)}")

    # (11) Cleanup
    if accelerator.is_main_process:
        os.system("rm -rf outputs_multi/checkpoint-*")
        logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    # Set up logging
    logger.add("training.log", rotation="500 MB")
    
    context_length = 8192
    current_folder = os.path.dirname(os.path.realpath(__file__))
    
    # Load training args with proper error handling
    try:
        with open(f"{current_folder}/training_args.yaml", "r") as f:
            all_training_args = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load training args: {str(e)}")
        raise

    # Train all feasible models
    for model_id in all_training_args.keys():
        logger.info(f"Starting training for model {model_id}")
        try:
            train_lora(
                model_id=model_id,
                context_length=context_length,
                training_args=LoraTrainingArguments(**all_training_args[model_id]),
            )
        except Exception as e:
            logger.error(f"Training failed for {model_id}: {str(e)}")
            continue
