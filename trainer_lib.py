import abc
import collections
import dataclasses
import torch

from typing import Any, Optional

@dataclasses.dataclass
class TrainerConfig:
    data_file: str = "data.bin"
    batch_size: int = 12
    gradient_accumulation_steps: int = 40
    device: str = ("cuda" if torch.cuda.is_available() else "cpu")
    dtype: str = ("bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16")
    grad_clip: float = 1.0
    log_interval: int = 1
    max_num_steps: int = 600000
    seed: int = 0
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    init_lr: float = 6e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    num_warmup_steps: int = 2000
    num_lr_decay_steps: int = 600000
    min_lr: float = 6e-5

@dataclasses.dataclass
class TrainState:
    step: int
    model: collections.OrderedDict[str, torch.Tensor]
    optimizer: Any

class Trainer:

    @abc.abstractmethod
    def __init__(
        self,
        config: TrainerConfig,
        init_train_state: Optional[TrainState] = None
    ):
        """Initialize the trainer.

        This method should not perform any training.
        """
        pass

    @abc.abstractmethod
    def train(self) -> TrainState:
        """Train a model and return the state at the end of training."""
        pass

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        "Return the name of the trainer."
        pass
