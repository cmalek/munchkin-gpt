from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Dict, Literal, Any

from pydantic import ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

import torch
import numpy as np


#: A mapping from string names to PyTorch data types
DTYPE_MAPPING: Dict[str, torch.dtype] = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
}

NUMPY_DTYPE_MAPPING: Dict[str, Any] = {
    'uint16': np.uint16
}


class InferenceSettings(BaseSettings):
    """
    The default settings here are for GPT-2 124M on the OpenWebText dataset.
    """

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    # ==================
    # Model settings
    # ==================

    #: The number of layers in the model
    n_layer: int = 12
    #: The number of attention heads in the model
    n_head: int = 12
    #: The embedding dimensionality
    n_embd: int = 768
    #: The dropout rate.  0.0 means no dropout.  0.0 for pre-trained models,
    #: try 0.1+ for finetuning
    dropout: float = 0.0
    #: Should we use bias in the LayerNorm and Linear layers?
    bias: bool = False
    #: The block size, aka the context window size in tokens
    block_size: int = 1024

    # ==================
    # Misc settings
    # ==================

    #: The device to use.  Examples: ``cuda``, ``cpu``, ``cuda:0``, ``cuda:1``.
    #: On M1/M2 Macs, you can try setting this to ``mps``.
    device: str = 'cuda'
    #: The data type to use.  Examples: ``float32``, ``float16``. ``bfloat16``
    #: Note that ``bfloat16`` is not supported on all GPUs.
    dtype: str = 'bfloat16'

    @property
    def context(self) -> AbstractContextManager:
        """
        Return a context manager that performs autocasting if :py:attr:`dtype`
        is ``float16`` or ``bfloat16``.

        If :py:meth:`device_type` is ``cpu``, this returns a null context
        manager.

        Returns:
            A context manager
        """
        if self.device_type == 'cpu':
            return nullcontext()
        else:
            return torch.amp.autocast(
                device_type=self.device_type,
                dtype=self.torch_dtype
            )

    @property
    def device_type(self) -> str:
        """
        Return the device type (``cpu`` or ``cuda``) corresponding to :py:attr:`device`

        .. note::
            What about ``mps``?
        """
        return 'cuda' if 'cuda' in self.device else 'cpu'

    @property
    def torch_dtype(self) -> torch.dtype:
        """
        Return the PyTorch data type corresponding to :py:attr:`dtype`
        """
        return DTYPE_MAPPING[self.dtype]

    @field_validator('dtype')
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        if v == 'bfloat16':
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return v
            return 'float16'
        return v


class DatasetSettings(BaseSettings):
    """
    Settings for the dataset
    """

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    #: The dataset to use.  This should either be a dataset from the HuggingFace
    #: Datasets library, or one of ``shaekspeare_char`` or ``shakespeare``.
    dataset: str
    #: Dataset kwargs for the loader/tokenizer
    dataset_loader_kwargs: Dict[str, Any] = {}
    #: The size of the validation set, as a fraction of the training set
    val_size: float = 0.0005
    #: The numpy datatype to use for the dataset
    dataset_dtype: str = 'uint16'

    @property
    def output_dtype(self) -> torch.dtype:
        """
        Return the PyTorch data type to use for the dataset
        """
        return NUMPY_DTYPE_MAPPING[self.dataset_dtype]

    @property
    def output_dir(self) -> Path:
        """
        Return the output directory for the dataset and its artifacts.
        """
        return Path(__file__).parent.parent / 'etc/dataset' / self.dataset

    @property
    def train_path(self) -> Path:
        """
        Return the path to the training data
        """
        return self.output_dir / 'train.bin'

    @property
    def val_path(self) -> Path:
        """
        Return the path to the training data
        """
        return self.output_dir / 'val.bin'

    @property
    def meta_path(self) -> Path:
        """
        Return the path to the meta data.
        """
        return self.output_dir / 'meta.pkl'


class TrainingSettings(InferenceSettings, DatasetSettings):
    """
    The default settings here are for GPT-2 124M on the OpenWebText dataset.
    """
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    #: The name for this model
    name: str = 'gpt2'

    # ==================
    # I/O settings
    # ==================

    #: The interval at which to display evaulation stats, in iterations. We
    #: will also save a checkpoint at this interval, if
    #: :py:attr:`always_save_checkpoint` is ``True``
    eval_interval: int = 2000
    #: How often to log training stats
    log_interval: int = 1
    #: How often to evaluate the model, in iterations
    eval_iters: int = 200
    #: If ``True``, only evaluate the model, don't train it
    eval_only: bool = False
    #: The gradient accumulation steps.  Used to simulate larger batch sizes.
    gradient_accumulation_steps: int = 5 * 8
    #: The batch size: number of examples per iteration
    batch_size: int = 12
    #: The random seed for the RNGs
    seed: int = 1337
    #: Whether to save checkpoints even if the validation loss is worse
    always_save_checkpoint: bool = True
    #: Where to start our training from.  Valid values are: ``scratch``, ``resume``
    #: and ``gpt2-*`` (where ``*`` is the name of a GPT-2 model, e.g. ``gpt2-xl``)
    init_from: Literal[
        'scratch',
        'resume',
        'gpt2',
        'gpt2-medium',
        'gpt2-large',
        'gpt2-xl',
    ] = 'scratch'

    # ==================
    # WandB settings
    # ==================

    #: Whether to log to WandB
    wandb_log: bool = False
    #: The name of the WandB project to log to
    wandb_project: str = 'owt'
    #: The name of the WandB run
    wandb_run_name: str = 'gpt2'

    # ==================
    # Optimizer settings
    # ==================

    #: The maximum learning rate
    learning_rate: float = 6e-4
    #: The total number of iterations to train for
    max_iters: int = 600000
    #: The weight decay: a regularization parameter
    weight_decay: float = 1e-1
    #: The beta1 parameter for Adam
    beta1: float = 0.9
    #: The beta2 parameter for Adam
    beta2: float = 0.95
    #: Clip gradients to this value.  Set to 0.0 to disable
    grad_clip: float = 1.0

    # ============================
    # Learning rate decay settings
    # ============================

    #: Weather to decay the learning rate
    decay_lr: bool = True
    #: The number of iterations to warmup the learning rate for
    warmup_iters: int = 2000
    #: The number of iterations over which to decay the learning rate.
    #: Per Chinchilla, this should be approximately equal to :py:attr:`max_iters`
    lr_decay_iters: int = 600000
    #: The minimum learning rate
    min_lr: float = 6e-5

    # ========================================
    # Distributed Data Parallel (DDP) settings
    # ========================================

    #: Whether to use DDP
    ddp: bool = False
    #: The DDP backend to use.
    backend: Literal['nccl', 'gloo', 'mpi', 'ucc'] = 'nccl'
    #: DDP Rank
    ddp_rank: int = 0
    #: DDP Local Rank: what we use for this process
    ddp_local_rank: int = 0
    #: DDP World Size
    ddp_world_size: int = 1

    # ==================
    # Misc settings
    # ==================

    #: Whether to compile the model using PyTorch 1.9's new JIT compiler
    #: This makes the model faster.
    compile: bool = True

    @property
    def out_dir(self) -> Path:
        """
        Return the output directory for the model and its artifacts.
        """
        return Path(__file__).parent.parent / 'models' / self.name

    @property
    def checkpoint_path(self) -> Path:
        """
        Return the path to the checkpoint file
        """
        return Path(self.out_dir) / 'checkpoint.pt'

    @property
    def is_master_process(self) -> bool:
        """
        Return whether this is the master process, i.e. the process with
        rank 0.
        """
        return self.ddp_rank == 0

    @property
    def seed_offset(self) -> int:
        """
        Return the seed offset, i.e. the number of iterations we've trained
        for so far.  This is used to seed the RNGs.

        If we're not doing DDP, this is 0, otherwise it's the DDP rank.
        """
        if self.ddp:
            return self.ddp_rank
        return 0

    @property
    def random_seed(self) -> int:
        """
        Return the random seed, i.e. the seed to use for the RNGs.
        """
        return self.seed + self.seed_offset

    @property
    def tokens_per_iter(self) -> int:
        """
        Return the number of tokens per iteration, i.e. the gradient
        accumulation steps * batch size * the block size.

        If we're doing DDP, this is multiplied by the DDP world size.
        """
        n_tokens = self.gradient_accumulation_steps * self.batch_size * self.block_size * self.ddp_world_size
        if self.ddp:
            n_tokens *= self.ddp_world_size
        return n_tokens

    @model_validator(mode='after')  #: type: ignore
    def configure_device(self) -> 'TrainingSettings':
        """
        If we're doing DDP, set the device to be ``cuda:<ddp_local_rank>``,
        and divide the gradient accumulation steps by the DDP world size.

        Otherwise, do nothing.

        Raises:
            ValidationError: gradient_accumulation_steps must be evenly divisible by ddp_world_size

        Returns:
            An instance of :py:class:`TrainingSettings`
        """
        if self.ddp:
            self.device = f'cuda:{self.ddp_local_rank}'
            if not self.gradient_accumulation_steps % self.ddp_world_size == 0:
                raise ValidationError(
                    'For DDP: gradient_accumulation_steps must be evenly divisible by ddp_world_size'
                )
            self.gradient_accumulation_steps //= self.ddp_world_size
        return self
