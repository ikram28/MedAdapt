import os
#os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import sys
import logging
import argparse

import torch
import datasets
import transformers
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
import idr_torch
from transformers import DataCollatorForLanguageModeling

os.environ['WANDB_MODE'] = 'offline'
wandb_project = "SFT-BioMistral-7B-2"

if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

logger = logging.getLogger(__name__)

def main():
    # CLI argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", type=str, help="HuggingFace model name")
    parser.add_argument("--path_train_dataset", type=str, default="./ft-data/data/train_data.json")
    parser.add_argument("--path_eval_dataset", type=str, default="./ft-data/data/test_data.json")
    parser.add_argument("--output_dir", type=str, default="./SFT-MistralNachos-models/")
    parser.add_argument("--logging_dir", type=str, default="./SFT-MistralNachos-logs/")
    parser.add_argument("--epochs", type=int, default=5, required=True)
    parser.add_argument("--batch_size", type=int, default=4, required=True)
    parser.add_argument("--save_steps", type=int, default=100, required=True)
    parser.add_argument("--logging_steps", type=int, default=10, required=True)
    parser.add_argument("--seed", type=int, default=42, required=True)
    parser.add_argument("--learning_rate", type=float, default=2e-5, required=True)
    args = parser.parse_args()

    # Training Arguments
    training_args = transformers.TrainingArguments(
        bf16=True,
        do_eval=True,
        eval_strategy="epoch",     # <-- fixed name
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        push_to_hub=False,
        remove_unused_columns=True,
        report_to="wandb",
        save_strategy="steps",
        save_steps=args.save_steps,
        seed=args.seed,
        logging_dir=args.logging_dir,
        logging_first_step=True,
        group_by_length=True,
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        local_rank=idr_torch.local_rank,
    )

    set_seed(args.seed)

    # Logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")

    # Load data
    dataset = load_dataset('json', data_files={'train': args.path_train_dataset, 'test': args.path_eval_dataset})
    train_data = dataset['train']
    eval_data = dataset['test']

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 2048

    def tokenize_and_label(example):
        encoded = tokenizer(
            example["user_prompt"],
            padding="max_length",
            truncation=True,
            max_length=2048
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    train_data = train_data.map(tokenize_and_label, remove_columns=train_data.column_names)
    eval_data = eval_data.map(tokenize_and_label, remove_columns=eval_data.column_names)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    # Model
    logger.info("*** Load base model ***")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    # Align embeddings and pad id
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Resized model embeddings to {len(tokenizer)} tokens")

    # LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        modules_to_save=None,
        use_dora=True
    )

    # Trainer (drop the unsupported `tokenizer=` kwarg)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        data_collator=data_collator,
    )

    # Training
    train_result = trainer.train(resume_from_checkpoint=True)

    # Save results
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_model(training_args.output_dir)

    trainer.model.config.use_cache = True
    trainer.model.config.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
