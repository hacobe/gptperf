import argparse
import dataclasses
import time
import torch
import sys

import trainer_lib
import nanogpt_trainer_lib
import new_trainer_lib

_TRAINER_MODULE_REGISTRY = {
	nanogpt_trainer_lib.Trainer.name(): nanogpt_trainer_lib,
	new_trainer_lib.Trainer.name(): new_trainer_lib,
}

@dataclasses.dataclass
class Checkpoint:
	trainer: str
	config: trainer_lib.TrainerConfig
	train_state: trainer_lib.TrainState

def _add_trainer_arg(field: dataclasses.Field, help_str: str, parser: argparse.ArgumentParser):
	if field.type not in [bool, float, int, str]:
		raise ValueError(f"Unsupported Field type <{field.type}>")

	if field.type == bool:
		parser.add_argument(f"--{field.name}",
			action=argparse.BooleanOptionalAction,
			help=help_str)
		return

	parser.add_argument(f"--{field.name}",
		type=field.type,
		default=(None if isinstance(field.default, dataclasses._MISSING_TYPE) else field.default),
		help=help_str)

def _add_trainer_args(parser: argparse.ArgumentParser):
	# Add the flags from trainer_lib.TrainerConfig.
	field_names = set()
	for field in dataclasses.fields(trainer_lib.TrainerConfig):
		_add_trainer_arg(field, "See trainer_lib.TrainerConfig.", parser)
		field_names.add(field.name)

	# Add any other flags from the other TrainerConfigs.
	for trainer_module in _TRAINER_MODULE_REGISTRY.values():
		for field in dataclasses.fields(trainer_module.TrainerConfig):
			if field.name in field_names:
				continue
			_add_trainer_arg(field, f"See {trainer_module.__name__}.TrainerConfig.", parser)

def _read_checkpoint(checkpoint_file: str) -> Checkpoint:
	with open(checkpoint_file, "rb") as fin:
		checkpoint = torch.load(fin)
	trainer_module = _TRAINER_MODULE_REGISTRY[checkpoint["trainer"]]
	config = trainer_module.TrainerConfig(**checkpoint["config"])
	checkpoint = Checkpoint(
		trainer=checkpoint["trainer"],
		config=config,
		train_state=trainer_lib.TrainState(**checkpoint["train_state"])
	)
	return checkpoint

def _get_present_arg_names(argv) -> set:
	present_arg_names = set()
	for arg in argv[1:]:
		end = 0
		while end < len(arg) and arg[end] != "=":
			end += 1
		flag = arg[:end]
		assert flag.startswith("--")
		# argparse removes the no- prefix in args.
		start = 5 if flag.startswith("--no-") else 2
		present_arg_names.add(flag[start:])
	return present_arg_names

def main(args, present_arg_names):
	# Maybe read the initial checkpoint.
	trainer_module = _TRAINER_MODULE_REGISTRY[args.trainer]
	if args.init_checkpoint_file is not None:
		init_checkpoint = _read_checkpoint(args.init_checkpoint_file)
		config = init_checkpoint.config
		init_train_state = init_checkpoint.train_state
	else:
		config = trainer_module.TrainerConfig()
		init_train_state = None

	# Override config with command-line arguments.
	for arg_name in present_arg_names:
		setattr(config, arg_name, getattr(args, arg_name))

	trainer = trainer_module.Trainer(config, init_train_state)

	# Train.
	start_time = time.time()
	train_state = trainer.train()
	elapsed = time.time() - start_time

	# Write the final checkpoint.
	if args.final_checkpoint_file is not None:
		checkpoint = Checkpoint(
			trainer=args.trainer,
			config=config,
			train_state=train_state
		)
		with open(args.final_checkpoint_file, "wb") as fout:
			torch.save(dataclasses.asdict(checkpoint), fout)

	# Compare the actual checkpoint and the expected checkpoint.
	if args.expected_checkpoint_file is not None:
		# Note that we only compare the model weights.
		expected_checkpoint = _read_checkpoint(args.expected_checkpoint_file)
		if train_state.model.keys() != expected_checkpoint.train_state.model.keys():
			raise ValueError("Model keys do not match.")
		for key in expected_checkpoint.train_state.model.keys():
			if not torch.allclose(train_state.model[key], expected_checkpoint.train_state.model[key]):
				raise ValueError(f"Model tensor <{key}> does not match.")

	# Write the score.
	if args.score_file:
		with open(args.score_file, "w") as fout:
			fout.write(str(elapsed))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train.")
	parser.add_argument("--trainer", 
		choices=sorted(list(_TRAINER_MODULE_REGISTRY.keys())), required=True,
		help="Trainer.")	
	parser.add_argument("--init_checkpoint_file", 
		type=str, default=None,
		help="Path to the initial checkpoint file.")
	parser.add_argument("--final_checkpoint_file", 
		type=str, default=None,
		help="Path to the final checkpoint file.")
	parser.add_argument("--score_file", 
		type=str, default=None,
		help="Path to the score file.")
	parser.add_argument("--expected_checkpoint_file", 
		type=str, default=None,
		help="Path to the expected checkpoint file.")
	_add_trainer_args(parser)
	args = parser.parse_args()
	present_arg_names = _get_present_arg_names(sys.argv)
	main(args, present_arg_names)
