import trainer_lib

class Trainer(trainer_lib.Trainer):

	def __init__(
		self,
		config: trainer_lib.TrainerConfig,
		init_train_state: trainer_lib.TrainState
	):
		raise NotImplementedError

	def train(self) -> trainer_lib.TrainState:
		raise NotImplementedError
