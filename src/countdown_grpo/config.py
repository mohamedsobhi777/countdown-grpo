from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Config:
    model: str = "Qwen/Qwen3.5-0.8B-Base"
    model_revision: str = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
    dataset: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    dataset_revision: str = "408f70d177020686d34a56bba5952feb45aaaee4"
    output_dir: str = "outputs/countdown"
    seed: int = 42
    eval_fraction: float = 0.02
    eval_samples: int = 256
    train_samples: int = 20000
    max_steps: int = 300
    max_completion_length: int = 256
    num_generations: int = 8
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    learning_rate: float = 1e-5
    lora_rank: int = 16
    lora_alpha: int = 32
    save_steps: int = 50
    format_reward: float = 0.0

    def __post_init__(self):
        for name in ("eval_samples", "train_samples", "max_steps", "max_completion_length",
                     "num_generations", "per_device_train_batch_size",
                     "gradient_accumulation_steps", "lora_rank", "lora_alpha", "save_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_generations < 2:
            raise ValueError("GRPO requires at least two generations")
        batch = self.per_device_train_batch_size * self.gradient_accumulation_steps
        if batch % self.num_generations:
            raise ValueError("Effective batch size must be divisible by num_generations")
        if not 0 < self.eval_fraction < 1:
            raise ValueError("eval_fraction must be between zero and one")
        if not 0 <= self.format_reward <= 0.1:
            raise ValueError("format_reward must be between zero and 0.1")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between zero and one")

    def to_dict(self):
        return asdict(self)


def load_config(path):
    with Path(path).open("rb") as handle:
        return Config(**tomllib.load(handle))
