import argparse
import dataclasses
import json
import os
import time
import torch

import trainer_lib
import nanogpt_trainer_lib
import new_trainer_lib

_TRAINER_REGISTRY = {
	"nanogpt": nanogpt_trainer_lib.Trainer,
	"new": new_trainer_lib.Trainer,
}

_INIT_LR = 6e-4
_MAX_NUM_STEPS = 1
_DEFAULT_CONFIG = trainer_lib.TrainerConfig(
    data_file="train.bin",
    batch_size=12,
    gradient_accumulation_steps=40,
    device=("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=("bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"),
    grad_clip=1.0,
    log_interval=1,
    max_num_steps=_MAX_NUM_STEPS,
    seed=0,
    model=trainer_lib.ModelConfig(
        block_size=1024,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768
    ),
    optimizer=trainer_lib.OptimizerConfig(
        init_lr=_INIT_LR,
        weight_decay=1e-1,
        betas=(0.9, 0.95)
    ),
    learning_rate_scheduler=trainer_lib.LearningRateSchedulerConfig(
        init_lr=_INIT_LR,
        num_warmup_steps=2000,
        num_lr_decay_steps=600000,
        min_lr=6e-5
    )
)

def _read_checkpoint(checkpoint_file: str):
	with open(checkpoint_file, "rb") as fin:
		checkpoint = torch.load(fin)
	config = trainer_lib.TrainerConfig(**checkpoint["config"])
	config.model = trainer_lib.ModelConfig(**config.model)
	config.optimizer = trainer_lib.OptimizerConfig(**config.optimizer)
	config.learning_rate_scheduler = trainer_lib.LearningRateSchedulerConfig(
		**config.learning_rate_scheduler)

	init_train_state = trainer_lib.TrainState(
		step=checkpoint["step"],
		model=checkpoint["model"],
		optimizer=checkpoint["optimizer"]
	)	
	return config, init_train_state

def main(args):
	if args.init_checkpoint_file is not None:
		config, init_train_state = _read_checkpoint(args.init_checkpoint_file)
	else:
		config = _DEFAULT_CONFIG
		init_train_state = None

	# Override default config.

	if args.data_file is not None:
		config.data_file = args.data_file

	if args.max_num_steps is not None:
		config.max_num_steps = args.max_num_steps

	if args.batch_size is not None:
		config.batch_size = args.batch_size

	if args.gradient_accumulation_steps is not None:
		config.gradient_accumulation_steps = args.gradient_accumulation_steps

	if args.seed is not None:
		config.seed = args.seed

	if args.device is not None:
		config.device = args.device

	if args.small_model:
		config.model.block_size = 4
		config.model.n_layer = 2
		config.model.n_embd = 8
		config.model.n_head = 2

	# Train.
	trainer_cls = _TRAINER_REGISTRY[args.trainer]
	trainer = trainer_cls(config, init_train_state)

	start_time = time.time()
	train_state = trainer.train()
	elapsed = time.time() - start_time

	# Write checkpoint.

	if args.final_checkpoint_file is not None:
		checkpoint = {
			"config": dataclasses.asdict(config),
			"step": train_state.step,
			"model": train_state.model,
			"optimizer": train_state.optimizer
		}
		with open(args.final_checkpoint_file, "wb") as fout:
			torch.save(checkpoint, fout)

	# Compare actual and expected.

	if args.expected_checkpoint_file is not None:
		# Note that we do not compare the optimizer state.
		expected_config, expected_train_state = _read_checkpoint(args.expected_checkpoint_file)
		if config != expected_config:
			raise ValueError("Configs do not match.")
		if expected_train_state.model.keys() != train_state.model.keys():
			raise ValueError("Model keys do not match.")
		for key in expected_train_state.model.keys():
			if not torch.allclose(expected_train_state.model[key], train_state.model[key]):
				raise ValueError(f"Model tensor <{key}> does not match.")

	# Write score.

	if args.score_file:
		with open(args.score_file, "w") as fout:
			fout.write(str(elapsed))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train.")
	parser.add_argument("--trainer", 
		choices=sorted(list(_TRAINER_REGISTRY.keys())), required=True,
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

	# Override default config.
	parser.add_argument("--data_file",
		type=str, default=None,
		help="Path to the bin file to use for training.")
	parser.add_argument("--max_num_steps", 
		type=int, default=None,
		help="The maximum number of steps to train.")
	parser.add_argument("--batch_size", 
		type=int, default=None,
		help="Batch size.")
	parser.add_argument("--gradient_accumulation_steps", 
		type=int, default=None,
		help="Gradient accumulation steps.")
	parser.add_argument("--seed", 
		type=int, default=None,
		help="The random seed.")
	parser.add_argument("--device", 
		choices=["cpu", "cuda"], default=None,
		help="Device.")
	parser.add_argument("--small_model",
		action="store_true",
		help="Use a small model (useful for testing).")

	args = parser.parse_args()
	main(args)
