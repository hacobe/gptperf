import contextlib
import dataclasses
import inspect
import time
import torch
import torch.nn as nn
import numpy as np

import trainer_lib
import nanogpt_model_lib

from typing import Generator, Optional

@dataclasses.dataclass
class TrainerConfig(trainer_lib.TrainerConfig):
	flash_attn: bool = True
	fused_adamw: bool = True

def _create_optimizer(
	parameters: Generator[nn.Parameter, None, None],
	init_lr: float,
	weight_decay: float,
	beta1: float,
	beta2: float,
	fused_adamw: bool,
	device_type: str,
) -> torch.optim.Optimizer:
	params = [p for p in parameters if p.requires_grad]
	decay_params = [p for p in params if p.dim() >= 2]
	nodecay_params = [p for p in params if p.dim() < 2]
	optim_groups = [
		{"params": decay_params, "weight_decay": weight_decay},
		{"params": nodecay_params, "weight_decay": 0.0}
	]
	num_decay_params = sum(p.numel() for p in decay_params)
	num_nodecay_params = sum(p.numel() for p in nodecay_params)

	if fused_adamw and device_type != "cuda":
		raise ValueError("fused_adamw requires cuda device.")

	if fused_adamw and "fused" not in inspect.signature(torch.optim.AdamW).parameters:
		raise ValueError("fused_adamw unavailable.")

	extra_args = dict(fused=True) if fused_adamw else dict()

	optimizer = torch.optim.AdamW(
		optim_groups,
		lr=init_lr,
		betas=(beta1, beta2),
		**extra_args
	)
	return optimizer

def _get_lr(step, init_lr, min_lr, num_lr_decay_steps, num_warmup_steps):
    if step < num_warmup_steps:
        return init_lr * step / num_warmup_steps
    if step > num_lr_decay_steps:
        return min_lr
    decay_ratio = (step - num_warmup_steps) / (num_lr_decay_steps - num_warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (init_lr - min_lr)

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
		config: TrainerConfig,
		init_train_state: Optional[trainer_lib.TrainState] = None
	):
		self._config = config

		np.random.seed(config.seed)
		torch.manual_seed(config.seed)

		model_config = nanogpt_model_lib.ModelConfig(
		    block_size=config.block_size,
		    vocab_size=config.vocab_size,
		    n_layer=config.n_layer,
		    n_head=config.n_head,
		    n_embd=config.n_embd,
		    flash_attn=config.flash_attn,
		)
		self._model = nanogpt_model_lib.Model(model_config)


		if init_train_state is not None:
			self._model.load_state_dict(init_train_state.model)
		self._model.to(config.device)

		self._device_type = ("cuda" if "cuda" in config.device else "cpu")
		self._optimizer = _create_optimizer(
			parameters=self._model.parameters(),
			init_lr=config.init_lr,
			weight_decay=config.weight_decay,
			beta1=config.beta1,
			beta2=config.beta2,
			fused_adamw=config.fused_adamw,
			device_type=self._device_type
		)
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
		ctx = contextlib.nullcontext() if self._device_type == "cpu" else torch.amp.autocast(
			device_type=self._device_type, dtype=ptdtype)
		scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == "float16"))

		t0 = time.time()
		while self._step < config.max_num_steps:
			if self._step == 0:
				X, Y = _get_batch(
					config.data_file,
					config.batch_size,
					config.block_size,
					config.device
				)

			lr = _get_lr(
				step=self._step,
				init_lr=config.init_lr,
				min_lr=config.min_lr,
				num_lr_decay_steps=config.num_lr_decay_steps,
				num_warmup_steps=config.num_warmup_steps,
			)
			for param_group in self._optimizer.param_groups:
				param_group["lr"] = lr

			for micro_step in range(config.gradient_accumulation_steps):
				with ctx:
					logits, loss = self._model(X, Y)
					loss = loss / config.gradient_accumulation_steps 
				X, Y = _get_batch(
					config.data_file,
					config.batch_size,
					config.block_size,
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

	@staticmethod
	def name() -> str:
		return "nanogpt"