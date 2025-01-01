import os
TASK_ID = os.environ.get("TASK_ID", "10")
os.environ["WANDB_PROJECT"] = f"Flock_task_{TASK_ID}"
from dataclasses import dataclass

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
import wandb
import random

from dataset import SFTDataCollator, SFTDataset
from utils.constants import model2template


@dataclass
class LoraTrainingArguments:
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float


def train_lora(
    model_id: str, context_length: int, training_args: LoraTrainingArguments
):
    assert model_id in model2template, f"model_id {model_id} not supported"
    lora_config = LoraConfig(
        r=training_args.lora_rank,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "down_proj",
            "up_proj",
            "gate_proj",
            "ffn",
        ],
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        task_type="CAUSAL_LM",
    )

    # Load model in 4-bit to do qLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    training_args = SFTConfig(
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_steps=100,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="wandb",
        run_name=model_id,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        num_train_epochs=training_args.num_train_epochs,
        max_seq_length=context_length,
        metric_for_best_model="loss",
        load_best_model_at_end=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=os.environ["HF_TOKEN"],
    )

    # Load dataset
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

    additional_dataset = SFTDataset(
        file="data/demo_data.jsonl",
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
    )

    # Add my own generated dataset
    additional_data_size = len(additional_dataset.data_list) // 100
    additional_data_subset = random.sample(additional_dataset.data_list, additional_data_size)
    combined_data_list = train_dataset.data_list + additional_data_subset
    # Create a new dataset using the combined data list
    combined_dataset = SFTDataset(
        file="data/demo_data.jsonl",
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
        custom_data_list=combined_data_list,
    )

    # Define trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=combined_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        peft_config=lora_config,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=context_length),
    )

    # Train model
    print("Start to train the model.")
    trainer.train()
    # Evaluate model
    print("Start to validate the model")
    try:
        trainer.evaluate()
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        print("Continuing with the rest of the process...")

    # save model
    trainer.save_model("outputs")

    # remove checkpoint folder
    os.system("rm -rf outputs/checkpoint-*")

    wandb.finish()  # Finalize the WandB run
    # upload lora weights and tokenizer
    print("Training Completed.")

if __name__ == "__main__":
    
    # Define training arguments for LoRA fine-tuning
    training_args = LoraTrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )
    # Set model ID and context length
    model_id = "Qwen/Qwen1.5-0.5B"
    context_length = 2048
    # Start LoRA fine-tuning
    train_lora(
        model_id=model_id,
        context_length=context_length, 
        training_args=training_args
    )