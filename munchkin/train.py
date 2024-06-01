from dataclasses import asdict
from functools import cached_property
import inspect
import math
import os
import time
from typing import Any, Optional, Dict
import warnings

import click
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .settings import TrainingSettings
from .data import Dataset
from .model import GPT, GPTModelLoader
from .utils import pprint_dict


warnings.filterwarnings("ignore", category=UserWarning)


class OptimizerFactory:
    """
    This class encapsulates the logic for creating our AdamW optimizer
    with the configuration we need.
    """

    @staticmethod
    def new(
        settings: TrainingSettings,
        model: GPT,
        checkpoint: Optional[Any] = None
    ) -> torch.optim.AdamW:
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in model.named_parameters()}
        # Filter out those that do not require gradients
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Create optimizer groups.  Any parameter that is 2D will be weight
        # decayed, otherwise no.  i.e. all weight tensors in matmuls +
        # embeddings decay, all biases and LayerNorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": settings.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and settings.device_type == "cuda"
        extra_args = {'fused': True} if use_fused else {}
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=settings.learning_rate,
            betas=(
                settings.beta1,
                settings.beta2,
            ),
            **extra_args
        )
        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        pprint_dict(
            {
                'Decayed parameters': f"{len(decay_params)}, with {num_decay_params:,} parameters",
                'Non-decayed parameters': f"{len(nodecay_params)}, with {num_nodecay_params:,} parameters",
                'Betas': f'({settings.beta1}, {settings.beta2})',
                'Using fused AdamW': use_fused,
            },
            title='Optimizer',
            align=30
        )
        return optimizer


class GPTEvaluator:
    """
    A class that evaluates a GPT model.  We use this to monitor how our model
    training is going.

    To use:

    .. code-block:: python

        settings = EvaluationSettings(env_file='/path/to/evaluate-config.env')
        evaluator = GPTEvaluator(settings)
        evaluator.evaluate()

    Args:
        settings: The evaluation settings
    """
    def __init__(self, settings: TrainingSettings):
        #: The training settings
        self.settings = settings
        #: The dataset
        self.dataset: Dataset = Dataset(settings)
        #: The vocabulary size
        self.vocab_size = self.dataset.vocab_size

    @cached_property
    def checkpoint(self) -> Optional[Any]:
        """
        Return the path to the checkpoint file.
        """
        if self.settings.checkpoint_path.exists():
            checkpoint = torch.load(self.settings.checkpoint_path, map_location=self.settings.device)
            return checkpoint
        return None

    @cached_property
    def model(self) -> GPT:
        """
        Return the model.
        """
        model = GPTModelLoader().new(
            self.settings,
            init_from=self.settings.init_from,
            checkpoint=self.checkpoint,
            vocab_size=self.vocab_size
        )
        if self.settings.compile:
            # compile the model, if desired
            model = torch.compile(model, backend='aot_eager')  # type: ignore
        return model

    def setup(self) -> None:
        """
        Set us up for evaluation.
        """
        # Set up torch itself
        torch.manual_seed(self.settings.random_seed)
        # Allow TF32 on matmul and cudnn
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set up the model
        self.model  # pylint: disable=pointless-statement

    @torch.no_grad()
    def estimate_loss(self, progress: bool = False) -> Dict[str, float]:
        """
        Estimate our loss on the training and validation sets, cross-entropy
        loss function.

        Returns:
            A dictionary with the training and validation losses.
        """
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.settings.eval_iters)
            if progress:
                r = tqdm(
                    range(self.settings.eval_iters),
                    desc=f"Estimating {split} loss",
                    colour='#777777'
                )
            else:
                r = range(self.settings.eval_iters)
            for k in r:
                inputs, targets = self.dataset.get_batch(split, self.settings)  # type: ignore
                with self.settings.context:
                    _, loss = self.model(inputs, targets)
                losses[k] = loss.item()
            out[split] = float(losses.mean())
        return out


class GPTTrainer(GPTEvaluator):
    """
    A class that encapsulates the training process for a GPT model.

    To use:

    .. code-block:: python

        settings = TrainingSettings(env_file='/path/to/training-config.env')
        trainer = GPTTrainer(settings)
        trainer.train()

    Args:
        settings: The training settings
    """

    def __init__(self, settings: TrainingSettings):
        super().__init__(settings)
        #: Our iteration counter
        self.iter_num: int = 0
        #: Our iteration counter for this session.  This may be different from
        #: :attr:`iter_num` if we are resuming from a checkpoint.
        self.local_iter_num: int = 0
        #: The best validation loss we've seen so far
        self.best_val_loss: float = 1e9
        #: Our model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.
        self.mfu: float = -1.0
        #: Save the raw model here
        self.raw_model: Optional[GPT] = None

    @property
    def checkpoint(self) -> Optional[Any]:
        """
        Return the path to the checkpoint file.
        """
        checkpoint = super().checkpoint
        if (
            checkpoint is not None and
            self.settings.init_from == 'resume'
        ):
            self.iter_num = checkpoint["iter_num"]
            self.best_val_loss = checkpoint["best_val_loss"]
        return checkpoint

    def setup(self) -> None:
        """
        Set up the training process.
        """
        info = {
            'Device': self.settings.device,
            'Parameter data type': self.settings.dtype,
            'Tokens per iteration': self.settings.tokens_per_iter,
            'Random seed': self.settings.random_seed,
            'Using DDP': self.settings.ddp,
        }

        # Set up DDP, if desired
        if self.settings.ddp:
            info['DDP backend'] = self.settings.backend
            info['DDP local rank'] = self.settings.ddp_local_rank
            torch.distributed.init_process_group(
                backend=self.settings.backend,
                world_size=self.settings.ddp_world_size,
                rank=self.settings.ddp_rank,
            )

        # Set up torch itself
        torch.manual_seed(self.settings.random_seed)
        # Allow TF32 on matmul and cudnn
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if self.settings.is_master_process:
            info['Checkpoint file'] = self.settings.checkpoint_path
            os.makedirs(self.settings.out_dir, exist_ok=True)

        # Set up the model
        self.model  # pylint: disable=pointless-statement
        info['GPT parameters'] = self.model.get_num_params()

        pprint_dict(info, title='Training Setup', align=30)
        # Set up the optimizer
        self.optimizer  # pylint: disable=pointless-statement
        # Set up the scaler
        self.scaler  # pylint: disable=pointless-statement

    @cached_property
    def model(self) -> GPT:
        """
        Return the model.
        """
        model = super().model
        self.raw_model = model
        if self.settings.ddp:
            # set up distributed data parallelism, if desired
            model = DDP(model, device_ids=[self.settings.ddp_local_rank])  # type: ignore
            self.raw_model = model.module  # type: ignore
        return model

    @cached_property
    def scaler(self) -> torch.cuda.amp.GradScaler:
        """
        Return the scaler for mixed precision training.

        This is only enabled if the dtype is float16.
        """
        return torch.cuda.amp.GradScaler(
            enabled=(self.settings.dtype == 'float16')
        )

    @cached_property
    def optimizer(self) -> torch.optim.AdamW:
        """
        Configure the AdamW optimizers.   We use this to update the weights of
        the model during training.

        Returns:
            The configured AdamW optimizer.
        """
        return OptimizerFactory.new(
            self.settings,
            self.model,
            self.checkpoint
        )

    @property
    def learning_rate(self) -> float:
        """
        Return the current learning rate.

        If :attr:`TrainingSettings.decay_lr` is ``False``, this is just the
        learning rate from the settings.  Otherwise, we decay the learning rate
        according to a schedule.

        Returns:
            The current learning rate.
        """
        if not self.settings.decay_lr:
            return self.settings.learning_rate
        rate: float = 0.0
        if self.iter_num < self.settings.warmup_iters:
            # Linearly increaase learning rate during the warmup period
            rate = self.settings.learning_rate * self.iter_num / self.settings.warmup_iters
        elif self.iter_num > self.settings.lr_decay_iters:
            # Don't decay past the end of the schedule
            rate = self.settings.min_lr
        else:
            # Cosine decay learning rate down to min_lr
            decay_ratio: float = float(self.iter_num - self.settings.warmup_iters)
            decay_ratio /= self.settings.lr_decay_iters - self.settings.warmup_iters
            assert 0 <= decay_ratio <= 1, f"decay_ratio is out of bounds: {decay_ratio}"
            # coeff range: [0,1]
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            rate = self.settings.min_lr + coeff * (self.settings.learning_rate - self.settings.min_lr)
        return rate

    @torch.no_grad()
    def estimate_loss(self, progress: bool = False) -> Dict[str, float]:
        """
        Estimate our loss on the training and validation sets, cross-entropy
        loss function.

        Returns:
            A dictionary with the training and validation losses.
        """
        out = super().estimate_loss(progress=True)
        self.model.train()
        return out

    def save_checkpoint(self) -> None:
        """
        Save a checkpoint for our model
        """
        assert self.raw_model is not None, "save_checkpoint called before model is initialized"
        if self.settings.is_master_process and self.iter_num > 0:
            checkpoint = {
                "model": self.raw_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "model_args": asdict(self.raw_model.config),
                "iter_num": self.iter_num,
                "best_val_loss": self.best_val_loss,
                "config": self.settings.model_dump(),
            }
            torch.save(checkpoint, self.settings.checkpoint_path)
            click.secho(f"Saved checkpoint to {self.settings.checkpoint_path}", fg='yellow')

    def evaluate(self) -> None:
        """
        Evaluate our model on the training and validation sets, and write
        checkpoints, if desired.

        We only evaluate every :py:attr:`TrainingSettings.eval_interval`
        iterations, and only if we are the only process or we are the master
        process in a distributed training run.

        Side Effects:
            If the validation loss is the best we've seen so far, we write a
            checkpoint, and update :py:attr:`best_val_loss`.
        """
        if self.iter_num % self.settings.eval_interval == 0 and self.settings.is_master_process:
            losses = self.estimate_loss()
            fg = 'red' if losses['val'] > self.best_val_loss else 'green'
            click.secho(
                f"Eval {self.iter_num}: train loss: {losses['train']:.4f}, "
                f"val loss: {losses['val']:.4f}",
                fg=fg
            )
            if losses['val'] < self.best_val_loss or self.settings.always_save_checkpoint:
                self.best_val_loss = losses['val']
                self.save_checkpoint()

    def log(self, loss: torch.Tensor, elapsed: float, deltaT: float) -> None:
        """
        Log some status information every so often.

        We output a log every :py:attr:`TrainingSettings.log_interval`
        iterations, if we are the only process or we are the master process in a
        distributed training run.

        Args:
            loss: The loss for the last micro batch
            elapsed: The time elapsed since the start of training
            deltaT: The time for the last micro batch
        """
        assert self.raw_model is not None, "log called before model is initialized"
        if self.iter_num % self.settings.log_interval == 0 and self.settings.is_master_process:
            lossf = loss.item() * self.settings.gradient_accumulation_steps
            # Let the training loop settle a bit before we start logging
            if self.local_iter_num >= 5:
                mfu = self.raw_model.estimate_mfu(
                    self.settings.batch_size * self.settings.gradient_accumulation_steps,
                    deltaT
                )
                self.mfu = mfu if self.mfu == -1.0 else 0.9 * self.mfu + 0.1 * mfu
                click.secho(
                    f"Iter {self.iter_num}: loss={lossf:.4f}, elapsed={elapsed:.2f} per_iter={deltaT:.2f}s, "
                    f"mfu {self.mfu * 100:.2f}%  lr={self.learning_rate:.2e}",
                )

    def train(self) -> None:
        """
        The training loop.
        """
        self.setup()
        click.secho('\n\nTraining', fg='yellow')
        click.secho('========', fg='yellow')
        _input, targets = self.dataset.get_batch("train", self.settings)
        t_start = time.time()
        t0 = t_start
        while self.iter_num <= self.settings.max_iters:
            # Set the learning rate for this iteration
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate
            # Evaluate the model on the training and validation sets
            self.evaluate()
            if self.iter_num == 0 and self.settings.eval_only:
                # If we're only evaluating, we're done
                break
            for micro_step in range(self.settings.gradient_accumulation_steps):
                if self.settings.ddp:
                    # In DDP training, we only need to sync gradients at the
                    # last micro step.  The official way to do this is with
                    # model.no_sync() context manager, but I really dislike that
                    # this bloats the code and forces us to repeat code looking
                    # at the source of that context manager, it just toggles
                    # this variable -- Andrej Karpathy
                    self.model.require_backward_grad_sync = False  # type: ignore
                    if micro_step == self.settings.gradient_accumulation_steps - 1:
                        self.model.require_backward_grad_sync = True  # type: ignore
                with self.settings.context:
                    _, loss = self.model(_input, targets)
                    # scale the loss to account for gradient accumulation
                    loss = loss / self.settings.gradient_accumulation_steps
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                _input, targets = self.dataset.get_batch("train", self.settings)
                # backward pass, with gradient scaling if training in fp16
                self.scaler.scale(loss).backward()

            # Clip gradients, update weights
            if self.settings.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # Flush out gradients, no need to keep them around
            self.optimizer.zero_grad(set_to_none=True)
            # loss here is the loss of the last micro batch, scaled
            t1 = time.time()
            self.log(loss, t1 - t_start, t1 - t0)
            t0 = t1
            # update iteration counters
            self.iter_num += 1
            self.local_iter_num += 1

        if self.settings.ddp:
            # tear down distributed data parallelism
            torch.distributed.destroy_process_group()
