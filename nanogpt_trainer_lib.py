import contextlib
import time
import torch
import numpy as np

import trainer_lib
import nanogpt_model_lib

from typing import Optional

def _create_optimizer(parameters, config: trainer_lib.OptimizerConfig) -> torch.optim.Optimizer:
	params = [p for p in parameters if p.requires_grad]
	decay_params = [p for p in params if p.dim() >= 2]
	nodecay_params = [p for p in params if p.dim() < 2]
	optim_groups = [
		{"params": decay_params, "weight_decay": config.weight_decay},
		{"params": nodecay_params, "weight_decay": 0.0}
	]
	num_decay_params = sum(p.numel() for p in decay_params)
	num_nodecay_params = sum(p.numel() for p in nodecay_params)
	optimizer = torch.optim.AdamW(
		optim_groups,
		lr=config.init_lr,
		betas=config.betas)
	return optimizer

def _get_lr(step, config: trainer_lib.LearningRateSchedulerConfig):
    if step < config.num_warmup_steps:
        return config.init_lr * step / config.num_warmup_steps
    if step > config.num_lr_decay_steps:
        return config.min_lr
    decay_ratio = (step - config.num_warmup_steps) / (config.num_lr_decay_steps - config.num_warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * config.decay_ratio))
    return min_lr + coeff * (config.init_lr - config.min_lr)

def _get_batch(data_file, batch_size, block_size, device):
	data = np.memmap(data_file, dtype=np.uint16, mode='r')
	ix = torch.randint(len(data) - block_size, (batch_size,))
	x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
	y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
	x, y = x.to(device), y.to(device)
	return x, y

class Trainer(trainer_lib.Trainer):

	def __init__(
		self,
		config: trainer_lib.TrainerConfig,
		init_train_state: Optional[trainer_lib.TrainState] = None
	):
		self._config = config

		np.random.seed(config.seed)
		torch.manual_seed(config.seed)

		self._model = nanogpt_model_lib.Model(config.model)
		if init_train_state is not None:
			self._model.load_state_dict(init_train_state.model)
		self._model.to(config.device)

		self._optimizer = _create_optimizer(self._model.parameters(), config.optimizer)
		if init_train_state is not None:
			self._optimizer.load_state_dict(init_train_state.optimizer)

		self._step = 0
		if init_train_state is not None:
			self._step = init_train_state.step

	def train(self) -> trainer_lib.TrainState:
		config = self._config

		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True
		ptdtype = {
			"float32": torch.float32,
			"bfloat16": torch.bfloat16,
			"float16": torch.float16
		}[config.dtype]
		device_type = "cuda" if "cuda" in config.device else "cpu"
		ctx = contextlib.nullcontext() if device_type == "cpu" else torch.amp.autocast(
			device_type=device_type, dtype=ptdtype)
		scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == "float16"))

		t0 = time.time()
		while self._step < config.max_num_steps:
			if self._step == 0:
				X, Y = _get_batch(
					config.data_file,
					config.batch_size,
					config.model.block_size,
					config.device
				)

			lr = _get_lr(self._step, config.learning_rate_scheduler)
			for param_group in self._optimizer.param_groups:
				param_group["lr"] = lr

			for micro_step in range(config.gradient_accumulation_steps):
				with ctx:
					logits, loss = self._model(X, Y)
					loss = loss / config.gradient_accumulation_steps 
				X, Y = _get_batch(
					config.data_file,
					config.batch_size,
					config.model.block_size,
					config.device
				)
				scaler.scale(loss).backward()

			if config.grad_clip != 0.0:
				scaler.unscale_(self._optimizer)
				torch.nn.utils.clip_grad_norm_(self._model.parameters(), config.grad_clip)
			scaler.step(self._optimizer)
			scaler.update()
			self._optimizer.zero_grad(set_to_none=True)

			t1 = time.time()
			dt = t1 - t0
			t0 = t1
			if self._step % config.log_interval == 0:
				lossf = loss.item() * config.gradient_accumulation_steps
				print(f"step {self._step}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

			self._step += 1

		return trainer_lib.TrainState(
			step=self._step,
			model=self._model.state_dict(),
			optimizer=self._optimizer.state_dict()
		)
