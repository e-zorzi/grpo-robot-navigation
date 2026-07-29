from dataclasses import dataclass


@dataclass
class GRPOConfig:
    """Configuration for Qwen VL GRPO training (robot navigation task)."""

    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    """HF checkpoint (repo id or local path) to fine-tune."""

    evaluator_model_name: str = "gpt-oss-120b"
    """Model used to score/evaluate generations for the reward function."""

    output_dir: str = "./qwen_grpo_robot"
    """Directory to save checkpoints and the final model."""

    dataset_name: str = "reasoning-augmentation/rubrics"
    """HF dataset repo id (or local path) to train on."""

    dataset_split: str = "train"
    """Dataset split to use."""

    alpha: float = 0.5
    """Reward mixing coefficient (component 1)."""

    beta: float = 0.3
    """Reward mixing coefficient (component 2)."""

    gamma: float = 0.2
    """Reward mixing coefficient (component 3)."""

    lora_r: int = 128
    """LoRA rank."""

    lora_alpha: int = 64
    """LoRA alpha."""

    num_epochs: int = 1
    """Number of training epochs."""

    batch_size: int = 4
    """Nominal/global batch size (informational; trainer uses per_device_train_batch_size)."""

    per_device_train_batch_size: int = 4
    """Per-device batch size passed to GRPOTrainer."""

    learning_rate: float = 1e-6
    """Learning rate."""

    max_completion_length: int = 256
    """Max number of tokens generated per completion."""

    num_generations: int = 4
    """Number of completions sampled per prompt (GRPO group size)."""

    use_vllm: bool = False
    """Whether to use vLLM for rollout generation."""

    save_steps: int = 2000
    """Checkpoint save interval, in steps."""

    logging_steps: int = 10
    """Logging interval, in steps."""

    device_num: str = 0
    """Device number (CUDA)"""

    save_artifacts: bool = False
    """Wheter to save the output as artifact on Wandb or not"""