"""
QLoRA fine-tuning for FinSight.

Technique: QLoRA (4-bit base model + LoRA adapters) via HuggingFace PEFT.
Tracks: perplexity, train/eval loss to MLflow.

Usage:
    python -m training.train --model microsoft/phi-3-mini-4k-instruct
    python -m training.train --model meta-llama/Llama-3.2-3B-Instruct --epochs 3
"""

import argparse
import logging
import math
import os
from pathlib import Path

import mlflow
import torch
from datasets import load_from_disk
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from training.dataset import prepare_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for FinSight")
    parser.add_argument("--model", default="microsoft/phi-3-mini-4k-instruct")
    parser.add_argument("--output_dir", default="models/finsight-adapter")
    parser.add_argument("--dataset_dir", default="data/finsight_dataset")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--mlflow_uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization (for CPU/MPS)")
    return parser.parse_args()


def compute_perplexity(trainer, dataset) -> float:
    metrics = trainer.evaluate(eval_dataset=dataset)
    return math.exp(metrics["eval_loss"])


def main():
    args = parse_args()

    # MLflow setup
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("finsight-finetuning")

    # Dataset
    dataset_path = Path(args.dataset_dir)
    if dataset_path.exists():
        logger.info(f"Loading cached dataset from {dataset_path}")
        split = load_from_disk(str(dataset_path))
    else:
        logger.info("Preparing dataset from scratch...")
        split = prepare_dataset(args.model, output_dir=args.dataset_dir)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
        )

    train_ds = split["train"].map(tokenize, batched=True, remove_columns=["text"])
    eval_ds = split["test"].map(tokenize, batched=True, remove_columns=["text"])

    # Model with QLoRA
    use_4bit = not args.no_4bit and torch.cuda.is_available()
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Using 4-bit QLoRA quantization")
    else:
        logger.info("Running in full precision (CPU/MPS or --no_4bit flag)")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto" if use_4bit else None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not use_4bit else None,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention layers
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=use_4bit,
        bf16=not use_4bit and torch.cuda.is_available(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",  # We handle MLflow manually
        dataloader_num_workers=0,
        group_by_length=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    # Baseline perplexity (before fine-tuning)
    logger.info("Computing baseline perplexity...")
    baseline_ppl = compute_perplexity(trainer, eval_ds)
    logger.info(f"Baseline perplexity: {baseline_ppl:.2f}")

    # Train
    with mlflow.start_run(run_name=f"finsight-{args.model.split('/')[-1]}"):
        mlflow.log_params({
            "model": args.model,
            "epochs": args.epochs,
            "lr": args.lr,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "batch_size": args.batch_size * args.grad_accum,
            "quantization": "4bit-QLoRA" if use_4bit else "full-precision",
        })
        mlflow.log_metric("baseline_perplexity", baseline_ppl)

        logger.info("Starting fine-tuning...")
        train_result = trainer.train()

        # Log training metrics
        for key, val in train_result.metrics.items():
            mlflow.log_metric(key, val)

        # Final perplexity
        final_ppl = compute_perplexity(trainer, eval_ds)
        logger.info(f"Final perplexity: {final_ppl:.2f} (was {baseline_ppl:.2f})")
        mlflow.log_metric("final_perplexity", final_ppl)
        mlflow.log_metric("perplexity_reduction_pct", (baseline_ppl - final_ppl) / baseline_ppl * 100)

        # Save adapter
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        mlflow.log_artifacts(args.output_dir, artifact_path="adapter")
        logger.info(f"Adapter saved to {args.output_dir}")

    print(f"\n{'='*50}")
    print(f"Fine-tuning complete!")
    print(f"  Baseline perplexity : {baseline_ppl:.2f}")
    print(f"  Final perplexity    : {final_ppl:.2f}")
    print(f"  Reduction           : {(baseline_ppl - final_ppl) / baseline_ppl * 100:.1f}%")
    print(f"  Adapter saved to    : {args.output_dir}")
    print(f"{'='*50}\n")
    print(f"To serve: set ADAPTER_PATH={args.output_dir} in .env and run docker compose up")


if __name__ == "__main__":
    main()
