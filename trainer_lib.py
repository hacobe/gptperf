import abc
import collections
import dataclasses
import torch

from typing import Any, Optional

@dataclasses.dataclass
class TrainState:
	step: int
	model: collections.OrderedDict[str, torch.Tensor]
	optimizer: Any

@dataclasses.dataclass
class ModelConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int

@dataclasses.dataclass
class OptimizerConfig:
    init_lr: float
    weight_decay: float
    betas: tuple[float, float]

@dataclasses.dataclass
class LearningRateSchedulerConfig:
    init_lr: float
    num_warmup_steps: int
    num_lr_decay_steps: int
    min_lr: float

@dataclasses.dataclass
class TrainerConfig:
    data_file: str
    batch_size: int
    gradient_accumulation_steps: int
    device: str
    dtype: str
    grad_clip: float
    log_interval: int
    max_num_steps: int
    seed: int

    model: ModelConfig
    optimizer: OptimizerConfig
    learning_rate_scheduler: LearningRateSchedulerConfig

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
