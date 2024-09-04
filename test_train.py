import pytest
import re
import numpy as np
import os
import tempfile
import torch
import subprocess

_PATTERN = r'step (\d+): loss ([\d.]+)'

# Expected losses were computed on "gpu_1x_a100_sxm4" instance using Lambda Labs.
# That instance has the following specs:
#
# 1x A100 (40 GB SXM4)
# 30 CPU cores, 205.4 GB RAM, 525.8 GB SSD
#
# Here's the nanoGPT setup:
#
# git clone git@github.com:karpathy/nanoGPT.git
# cd nanoGPT
# git checkout 9755682
#
# Replace `data_dir = os.path.join('data', dataset)` on line 115 with `data_dir = ""`.
# Replace `train.bin` with `data.bin` on line 120.
# Add `torch.manual_seed(123)` before `gptconf = GPTConfig(**model_args)` on line 156.

# python train.py \
# 	--compile=False \
#	--device=cpu \
# 	--batch_size=2 \
# 	--gradient_accumulation_steps=2 \
# 	--block_size=4 \
# 	--n_layer=2 \
# 	--n_embd=8 \
# 	--n_head=2 \
# 	--max_iters=4
_EXPECTED_LOSSES_CPU = [10.8171, 10.8665, 10.8036, 10.8445, 10.8278]

# python train.py \
# 	--compile=False \
# 	--batch_size=2 \
# 	--gradient_accumulation_steps=2 \
# 	--block_size=4 \
# 	--n_layer=2 \
# 	--n_embd=8 \
# 	--n_head=2 \
# 	--max_iters=4
_EXPECTED_LOSSES_CUDA = [10.8203, 10.8672, 10.8047, 10.8516, 10.8281]

def _run_command_and_assert_results_equal_to_nanogpt(args, expected_losses):
	result = subprocess.run(args, capture_output=True, text=True)
	assert result.returncode == 0
	lines = result.stdout.strip().split("\n")
	assert len(lines) == 5
	steps = []
	losses = []
	for line in lines:
		match = re.search(_PATTERN, line)
		assert match
		steps.append(int(match.group(1)))
		losses.append(float(match.group(2)))
	expected_steps = np.arange(len(expected_losses))
	assert (steps == expected_steps).all()
	np.testing.assert_almost_equal(np.array(losses), np.array(expected_losses))

@pytest.mark.parametrize("device, expected_losses", [
	("cpu", _EXPECTED_LOSSES_CPU),
	("cuda", _EXPECTED_LOSSES_CUDA)
])
def test_train_nanogpt_from_scratch(device, expected_losses):
	if device == "cuda" and not torch.cuda.is_available():
		pytest.skip()

	args = [
		"python",
		"train.py",
		"--trainer=nanogpt",
		"--data_file=data.bin",
		"--max_num_steps=5",
		"--batch_size=2",
		"--gradient_accumulation_steps=2",
		"--seed=123",
		"--block_size=4",
		"--n_layer=2",
		"--n_embd=8",
		"--n_head=2",
		f"--device={device}"
	]
	if device == "cpu":
		args.append("--no-fused_adamw")

	_run_command_and_assert_results_equal_to_nanogpt(args, expected_losses)

@pytest.mark.parametrize("device, expected_losses", [
	("cpu", _EXPECTED_LOSSES_CPU),
	("cuda", _EXPECTED_LOSSES_CUDA)
])
def test_train_nanogpt_from_checkpoint(device, expected_losses):
	if device == "cuda" and not torch.cuda.is_available():
		pytest.skip()

	with tempfile.TemporaryDirectory() as tmpdirname:
		checkpoint_file = os.path.join(tmpdirname, "small_checkpoint0.bin")
		args = [
			"python",
			"train.py",
			"--trainer=nanogpt",
			"--max_num_steps=0",
			"--seed=123",
			"--block_size=4",
			"--n_layer=2",
			"--n_embd=8",
			"--n_head=2",
			f"--device={device}",
			f"--final_checkpoint_file={checkpoint_file}"
		]
		if device == "cpu":
			args.append("--no-fused_adamw")
		result = subprocess.run(args, capture_output=True, text=True)
		assert result.returncode == 0

		args = [
			"python",
			"train.py",
			"--trainer=nanogpt",
			f"--init_checkpoint_file={checkpoint_file}",
			"--data_file=data.bin",
			"--max_num_steps=5",
			"--batch_size=2",
			"--gradient_accumulation_steps=2",
			f"--device={device}"
		]
		if device == "cpu":
			args.append("--no-fused_adamw")
		_run_command_and_assert_results_equal_to_nanogpt(args, expected_losses)
