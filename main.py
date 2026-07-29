import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_VERBOSITY"] = "warning"

import logging
import wandb

import torch
import tyro
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from trl import GRPOTrainer, GRPOConfig as TRLGRPOConfig
from peft import LoraConfig, get_peft_model
import functools
from config import GRPOConfig
from dataset import load_robot_dataset
from rewards import score_reward_func, format_reward_func, reasoning_reward_func

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)  # sometimes chatty too

logger = logging.getLogger(__name__)


def main(cfg: GRPOConfig) -> None:

    if cfg.save_artifacts:
        logger.info("> Will save artifacts on Wandb")
        wandb_artifact = wandb.Artifact(name="checkpoints", type="model")

    logger.info("=" * 60)
    logger.info("  Qwen VL GRPO Training - Robot Navigation")
    logger.info("=" * 60)
    logger.info(f"  Model:   {cfg.model_name}")
    logger.info(f"  Output:  {cfg.output_dir}")
    logger.info("=" * 60)
    logger.info("")

    logger.info("[1/5] Validating environment...")
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found!")
    gpu_name = torch.cuda.get_device_name(cfg.device_num)
    gpu_mem = torch.cuda.get_device_properties(cfg.device_num).total_memory // (2**20)
    logger.info(f"GPU: {gpu_name}")
    logger.info(f"GPU Memory: {gpu_mem:.1f} MiB")
    logger.info("Environment OK!")
    logger.info("")

    logger.info("[2/5] Loading dataset...")
    train_dataset = load_robot_dataset(cfg.dataset_name, cfg.dataset_split)
    logger.info(f"Dataset loaded! {len(train_dataset)} examples")
    logger.info("")

    logger.info("[3/5] Loading processor...")
    processor = AutoProcessor.from_pretrained(cfg.model_name)
    logger.info("Processor loaded!")
    logger.info("")

    logger.info("[4/5] Setting up GRPO trainer...")
    try:
        trl_config = TRLGRPOConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            num_generations=cfg.num_generations,
            max_completion_length=cfg.max_completion_length,
            learning_rate=cfg.learning_rate,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            save_total_limit=1,
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            use_vllm=cfg.use_vllm,
            report_to="wandb",
            reward_weights=[cfg.alpha, cfg.beta, cfg.gamma]
        )
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules="all-linear",
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model_name,
            torch_dtype="auto",
            device_map="auto",
            # attn_implementation="flash_attention_2",
        )

        model = get_peft_model(model, lora_config)
        print(model.print_trainable_parameters())

        trainer = GRPOTrainer(
            model=model,
            args=trl_config,
            train_dataset=train_dataset,
            processing_class=processor,
            reward_funcs=[format_reward_func, score_reward_func, reasoning_reward_func]
        )
        logger.info("Trainer ready!")
    except Exception as e:
        logger.error(f"Trainer setup failed: {e}")
        raise

    logger.info("[5/5] Starting training...")
    logger.info(f"  Epochs:       {cfg.num_epochs}")
    logger.info(f"  Batch size:   {cfg.batch_size}")
    logger.info(f"  LR:           {cfg.learning_rate}")
    logger.info(f"  Generations:  {cfg.num_generations}")
    logger.info("")

    try:
        trainer.train()
        logger.info("Training complete! ✅")
        trainer.save_model(cfg.output_dir)
        logger.info(f"Model saved to {cfg.output_dir}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        logger.info("Uploading artifacts on Wandb...")
        if cfg.save_artifacts:
            wandb_artifact.add_dir(cfg.output_dir)
            wandb.log_artifact(wandb_artifact)


if __name__ == "__main__":
    cfg = tyro.cli(GRPOConfig)
    main(cfg)