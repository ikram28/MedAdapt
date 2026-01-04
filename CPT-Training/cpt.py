import os
#os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import sys
import logging
import argparse

import wandb
import torch

import datasets
import transformers
from transformers import set_seed

from datasets import load_from_disk
from torch.distributed.elastic.multiprocessing.errors import record
#from transformers import AutoModelForCausalLM, set_seed, AutoTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM, set_seed
import idr_torch  
import torch.distributed as dist


# Disable online tracking by wandb (optional if working offline)
os.environ['WANDB_MODE'] = 'offline'

# Set the wandb project name
wandb_project = "XXX"
if wandb_project:
    os.environ["WANDB_PROJECT"] = wandb_project
logger = logging.getLogger(__name__)

def main():
    # For distributed training
    os.environ['LOCAL_RANK'] = os.environ.get('SLURM_LOCALID', '0')

    # Initialize distributed training using NCCL backend
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=idr_torch.size,
        rank=idr_torch.rank
    )

    print('start')

    # Define CLI arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", type=str, default="Mistral/Mistral", help="Model name from HuggingFace")
    parser.add_argument("--path_dataset", type=str, default="./dataset/Out/", help="Path to tokenized dataset")
    parser.add_argument("--output_dir", type=str, default="./Mistral-Base-Nachos/", help="Directory to save model")
    parser.add_argument("--logging_dir", type=str, default="./Mistral-base-logs/", help="Logging directory")
    parser.add_argument("--tokenizer_name",type=str)

    parser.add_argument("--epochs", type=int, default=3, required=True)
    parser.add_argument("--batch_size", type=int, default=32, required=True)
    parser.add_argument("--save_steps", type=int, default=80, required=True)
    parser.add_argument("--logging_steps", type=int, default=10, required=True)
    parser.add_argument("--seed", type=int, default=42, required=True)
    parser.add_argument("--learning_rate", type=float, default=2e-5, required=True)
    args = parser.parse_args()

    # Training Arguments 
    training_args = transformers.TrainingArguments(
        bf16=True,
        do_eval=False,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        push_to_hub=False,
        remove_unused_columns=True,
        report_to="wandb",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=20,
        seed=args.seed,
        #tf32=True,
        logging_dir=args.logging_dir,
        logging_first_step=True,
        optim="adamw_torch",
        fsdp="full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap='LlamaDecoderLayer',
        local_rank=idr_torch.local_rank,
        weight_decay=0.01,
        max_grad_norm=1.0,
    )

    # Reproducibility
    set_seed(training_args.seed)

    # Setup logging
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

    # Summary log
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bit training: {training_args.bf16}"
    )
    logger.info(f"Training arguments: {training_args}")

    # Load tokenizer
    #tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=2048)
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_name, model_max_length=2048)

    tokenizer.add_special_tokens({'pad_token': '<pad>'})  # Required for padding

    model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=None,
        torch_dtype=torch.bfloat16,
)
    model.resize_token_embeddings(len(tokenizer))  # <-- CRITICAL step in latest code


    # Load preprocessed/tokenized dataset
    tokenized_datasets = load_from_disk(args.path_dataset)

    model.gradient_checkpointing_enable()  

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Initialize trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_datasets,
        args=training_args,
        data_collator=data_collator,
    )

    model.config.use_cache = False  
    print("Tokenizer vocab size:", len(tokenizer))
    print("Model embedding shape:", model.get_input_embeddings().weight.shape)


    # Train 
    train_result = trainer.train()
    #train_result = trainer.train(resume_from_checkpoint=True)

    # Save metrics + model
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_model(training_args.output_dir)

    # Save config
    trainer.model.config.use_cache = True
    trainer.model.config.save_pretrained(training_args.output_dir)


def _mp_fn(index):
    main()

if __name__ == "__main__":
    main()
