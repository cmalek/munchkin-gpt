from contextlib import AbstractContextManager, nullcontext
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Dict

import click
import torch
from torch.nn import functional as F

from munchkin.utils import pprint_dict

from .data import Dataset
from .model import GPT, GPTModelLoader
from .settings import TrainingSettings


class GPTGenerator:
    """
    A class that encapsulates the text generation process for a GPT model.

    To use:

    .. code-block:: python

        settings = TrainingSettings(env_file='/path/to/training-config.env')
        generator = GPTGenerator(
            'model-name',
            device='mps',
            random_seed=23456,
            temperature=0.8,
            top_k=200,
        )
        generator.generate('here is a prompt')

    Args:
        settings: The training settings
    """

    def __init__(
        self,
        model_name: str,
        device: str = 'cuda',
        random_seed: int = 1337,
        temperature: float = 0.8,
        top_k: Optional[int] = None,
        compile: bool = False,
    ):
        #: The name of the model snapshot to use
        self.model_name = model_name
        #: The device to use for generation
        self.device = device
        #: The random seed to use for generation
        self.random_seed = random_seed
        #: The temperature to use for generation
        self.temperature = temperature
        #: The top_k to use for generation
        self.top_k = top_k
        #: Whether or not to compile the model
        self.compile = compile
        #: The encoding function to use
        self.encoder: Optional[Any] = None
        self.setup()

    def setup(self) -> None:
        """
        Set up the training process.
        """
        info = {
            'Device': self.device,
            'Random seed': self.random_seed,
            'Temperature': self.temperature,
            'Top k': self.top_k,
            'Compile': self.compile,
        }
        pprint_dict(info, title='Generation Setup', align=30)

        # Set up torch itself
        torch.manual_seed(self.random_seed)
        # Allow TF32 on matmul and cudnn
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Try to find the tokenizer
        config_path = Path(__file__).parent.parent / 'etc/training' / f'{self.model_name}.env'
        settings = TrainingSettings(_env_file=config_path)
        self.encoder = Dataset.get_tokenizer(settings.dataset)
        if hasattr(self.encoder, 'load_vocabulary'):
            self.encoder.load_vocabulary(settings.dataset)

        # Set up the model
        self.model  # pylint: disable=pointless-statement

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
        if self.device in ['cpu', 'mps']:
            return nullcontext()
        else:
            device = self.device.split(':')[0]
            return torch.amp.autocast(
                device_type=device,
                dtype=self.torch_dtype
            )

    @property
    def torch_dtype(self) -> torch.dtype:
        """
        Return the torch data type to use for the model.

        Returns:
            The torch data type
        """
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    @cached_property
    def checkpoint(self) -> Optional[Any]:
        """
        Return the path to the checkpoint file.
        """
        checkpoint_path = Path(__file__).parent.parent / 'models' / self.model_name / 'checkpoint.pt'
        assert checkpoint_path.exists(), f'No checkpoint found for "{self.model_name}"'
        return torch.load(checkpoint_path, map_location=self.device)

    @cached_property
    def model(self) -> GPT:
        """
        Return the model.
        """
        model: Any = GPT.from_checkpoint(self.checkpoint, dropout=0.0)
        model.eval()
        model.to(self.device)
        if self.compile:
            model = torch.compile(model)
        return model

    def tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenize the given text.
        """
        assert self.encoder is not None, 'No encoder found'
        idx = self.encoder.encode(text, allowed_special={'<|endoftext|>'})
        return torch.tensor(idx, dtype=torch.long, device=self.device)[None, ...]

    def untokenize(self, tokens: torch.Tensor) -> str:
        """
        Untokenize the given tokens.

        Args:
            tokens: the tensor of tokens to convert to text

        Returns:
            The text representation of the given tokens
        """
        assert self.encoder is not None, 'No encoder found'
        return self.encoder.decode(tokens[0].tolist())

    @torch.no_grad()
    def generate(
        self,
        text: str,
        max_new_tokens: int = 200,
    ) -> str:
        """
        This is the inference / generation function.  It takes a conditioning
        sequence of indices ``idx`` and generates ``max_new_tokens`` new tokens,
        feeding the predictions back into the model each time.  Most likely
        you'll want to make sure to be in model.eval() mode of operation for this.

        .. note::
            The ``torch.no_grad()`` decorator is used here to disable gradient
            calculation, which reduces memory consumption and speeds things up
            a bit.  This is because we don't need gradients for inference.

        Args:
            text: The text to use as a prompt

        Keyword Args:
            max_new_tokens: The maximum number of new tokens to generate.
        """
        idx = self.tokenize(text)
        with self.context:
            for _ in range(max_new_tokens):
                # if the sequence context is growing too long we must crop it at
                # block_size
                idx_cond = (
                    idx
                    if idx.size(1) <= self.model.config.block_size
                    else idx[:, -self.model.config.block_size:]
                )
                # Run our sequence through the model to get our logits
                logits, _ = self.model(idx_cond)

                # The rest here is the decoding of the logits to probabilities and
                # sampling from the distribution.
                # ---------------------------------------------------------------

                # Pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1, :] / self.temperature
                # Optionally crop the logits to only the top k options
                if self.top_k is not None:
                    v, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                # Apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # Random sample from the distribution to return the index of the
                # next token from the vocabulary
                idx_next = torch.multinomial(probs, num_samples=1)
                # Append that index to the running sequence and continue
                idx = torch.cat((idx, idx_next), dim=1)
        return self.untokenize(idx)
