import dataclasses
import trainer_lib

from typing import Optional

@dataclasses.dataclass
class TrainerConfig(trainer_lib.TrainerConfig):
	pass

class Trainer(trainer_lib.Trainer):

	def __init__(
		self,
		config: TrainerConfig,
		init_train_state: Optional[trainer_lib.TrainState] = None
	):
		raise NotImplementedError

	def train(self) -> trainer_lib.TrainState:
		raise NotImplementedError

	@staticmethod
	def name() -> str:
		return "new"
