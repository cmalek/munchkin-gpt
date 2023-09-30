"""
Full definition of a GPT Language Model, all of it in this single file.

References:

1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py

2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

"""
# mypy: disable-error-code="operator, union-attr"

from dataclasses import dataclass, asdict
import math
from typing import Optional, Dict, Any, cast

import click
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

from .settings import InferenceSettings
from .utils import pprint_dict



@dataclass
class GPTConfig:
    """
    This dataclass holds our hyperparameters for the architecture of the GPT model.
    """
    #: The context window size in tokens
    block_size: int = 1024
    #: The desired vocabulary size. This is the GPT-2 vocabulary size of 50257,
    #: padded up to nearest multiple of 64 for efficiency
    vocab_size: int = 50304
    #: The number of transformer blocks
    n_layer: int = 12
    #: The number of attention heads per transformer block
    n_head: int = 12
    #: The embedding dimensionality
    n_embd: int = 768
    #: The dropout probability
    dropout: float = 0.0
    #: Whether to include a bias in Linears and LayerNorms.  If ``True``, this is
    #: more like what GPT-2 does, but it is less memory efficient.  Setting to ``False``
    #: is an optimization that is a bit better and faster for training.
    bias: bool = True


class LayerNorm(nn.Module):
    """
    Our own implementation of :ref:`torch.nn.LayerNorm`, because in the PyTorch
    version, you can't specify the bias as None.

    Layer normalization is used in neural networks to normalize the activities
    of the neurons.  Performing this normalization helps the to reduce the
    training time of the model, and can improve the model's performance.

    The output of this is the input with a mean of zero and unit variance.


    Args:
        ndim: The number of dimensions of the input tensor.

    Keyword Arguments:
        bias: If ``True``, add a learnable bias to the normalized output.  Otherwise
            add no bias.

    """

    def __init__(self, ndim: int, bias: bool = False):
        super().__init__()
        #: The weight parameter.  Has shape ``(ndim,)``.
        self.weight: nn.Parameter = nn.Parameter(torch.ones(ndim))
        #: The bias parameter.  Has shape ``(ndim,)`` if ``bias=True``
        self.bias: Optional[nn.Parameter] = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        """
        Return ``input``, normalized by the mean and variance of its last 2
        dimensions.

        Args:
            input: The input tensor.  Should have shape ``(d0, d1, ..., dn, ndim)``.

        Returns:
            The normalized tensor, which has the same shape as the input.
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class AttentionHead(nn.Module):
    """
    Our self-attention layer. We use masked self-attention, so we also have a mask
    that prevents attention to future tokens.
    """

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        # Embedding dimensionality must be an integer multiple of the number of heads
        assert config.n_embd % config.n_head == 0
        #: key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        #: The output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        #: Dropouts for the attention layers
        self.attn_dropout = nn.Dropout(config.dropout)
        #: Dropout for the residual connections
        self.resid_dropout = nn.Dropout(config.dropout)
        #: The number of attention heads
        self.n_head = config.n_head
        #: The embedding dimensionality
        self.n_embd = config.n_embd
        #: The dropout rate.  We set this to 0 during evaluation, and > 0 during training.
        self.dropout = config.dropout
        #: Flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            click.secho(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0",
                fg='yellow'
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, input: torch.Tensor):
        """
        Do the self-attention computation

        Args:
            input: The input tensor.  Should have shape ``(batch_size,
                sequence_length, embedding_dimension)``.

        Returns:
            The output projection of the self-attention computation.
        """
        # batch size, sequence length, embedding dimensionality (n_embd)
        batch_size, seq_length, n_embed = input.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query, key, value = self.c_attn(input).split(self.n_embd, dim=2)

        # The keys: what we have
        key = key.view(
            batch_size,
            seq_length,
            self.n_head,
            n_embed // self.n_head
        ).transpose(1, 2)

        # The queries: what we want to find
        query = query.view(
            batch_size,
            seq_length,
            self.n_head,
            n_embed // self.n_head
        ).transpose(1, 2)

        # The values: what we will output based on what we find
        value = value.view(
            batch_size,
            seq_length,
            self.n_head,
            n_embed // self.n_head
        ).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.dropout,
                is_causal=True,
            )
        else:
            # Do the attention calculation
            att = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
            # Mask out our future tokens
            att = att.masked_fill(self.bias[:, :, :seq_length, :seq_length] == 0, float("-inf"))  # type: ignore
            # Turn the attention into a probability distribution
            att = F.softmax(att, dim=-1)
            # Maybe apply some attention dropout
            att = self.attn_dropout(att)
            y = att @ value  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_length, n_embed)

        # Apply dropout to the output of the attention layer
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """
    The multi-layer perceptron (MLP).  This is a simple, fully connected
    feed-forward network that is applied to each position separately and
    identically.

    Args:
        config: thee GPT configuration.
    """

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        #: The first linear layer of the perceptron
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        #: Our activation function: GELU
        self.gelu = nn.GELU()
        #: The second linear layer of the perceptron
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        #: Dropout for the output of the MLP
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input: torch.Tensor):
        """
        Process ``input``, a tensor of shape ``(batch_size, sequence_length,
        embedding_dimension)`` through the MLP.

        Args:
            input: The input tensor.  Should have shape ``(batch_size,
                sequence_length, embedding_dimension)``.
        """
        # Process the input through the first linear layer
        x = self.c_fc(input)
        # Apply the GELU activation function
        x = self.gelu(x)
        # Project the output back to the embedding dimensionality
        x = self.c_proj(x)
        #: Apply dropout to the output projection
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    A single transformer block.  It consists of two sub-layers. The first is a
    multi-head self-attention mechanism, and the second is a simple,
    position-wise fully connected feed-forward network.

    Args:
        config: The GPT model configuration.
    """

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        #: The input layer normalization
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        #: The self-attention head
        self.attn = AttentionHead(config)
        #: The output layer normalization
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        #: The feed-forward network
        self.mlp = MLP(config)

    def forward(self, input):
        """
        Process ``input``, a tensor of shape ``(batch_size, sequence_length,
        embedding_dimension)`` through the transformer block.
        """
        attended = input + self.attn(self.ln_1(input))
        return attended + self.mlp(self.ln_2(attended))


class GPT(nn.Module):

    PRETRAINED_MODEL_ARGS = {
        "gpt2": {'n_layer': 12, 'n_head': 12, 'n_embd': 768},          # 124M params
        "gpt2-medium": {'n_layer': 24, 'n_head': 16, 'n_embd': 1024},  # 350M params
        "gpt2-large": {'n_layer': 36, 'n_head': 20, 'n_embd': 1280},   # 774M params
        "gpt2-xl": {'n_layer': 48, 'n_head': 25, 'n_embd': 1600},      # 1558M params
    }

    @classmethod
    def from_checkpoint(cls, checkpoint: Any, dropout: Optional[float] = None) -> "GPT":
        dropout = checkpoint["model_args"]["dropout"] if dropout is None else dropout
        config = GPTConfig(
            n_layer=checkpoint['model_args']['n_layer'],
            n_head=checkpoint['model_args']['n_head'],
            n_embd=checkpoint['model_args']['n_embd'],
            block_size=checkpoint['model_args']['block_size'],
            dropout=dropout,
            bias=checkpoint['model_args']['bias'],
            vocab_size=checkpoint['model_args']['vocab_size'],
        )
        pprint_dict(asdict(config), title="Model configuration", align=30)

        model = cls(config)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :( honestly no idea how
        # checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k in state_dict:
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        click.secho("Loading state dict ... ", fg='green', nl=False)
        model.load_state_dict(state_dict)
        click.secho("Done", fg='cyan')
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_type: str,
        override_args: Optional[Dict[str, Any]] = None
    ) -> "GPT":
        assert model_type in cls.PRETRAINED_MODEL_ARGS, \
            f'unknown model type specified: {model_type}'
        if override_args is None:
            override_args = {}
        # n_layer, n_head and n_embd are determined from model_type
        config_args: Dict[str, Any] = {
            'vocab_size': 50257,  # always 50257 for GPT model checkpoints
            'block_size': 1024,   # always 1024 for GPT model checkpoints
            'bias': True,         # always True for GPT model checkpoints
        }
        config_args.update(cls.PRETRAINED_MODEL_ARGS[model_type])
        click.secho(
            "    WARNING: forcing vocab_size=50257, block_size=1024, bias=True",
            fg='yellow'
        )
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            click.secho(
                f"    WARNING: overriding dropout rate to {override_args['dropout']}",
                fg='yellow'
            )
            config_args["dropout"] = override_args["dropout"]
        click.secho(
            f"    Loading weights from pretrained gpt: {model_type}",
            fg='green'
        )

        # create a from-scratch initialized minGPT model
        model_config = GPTConfig(**config_args)
        model = GPT(model_config)
        sd = model.state_dict()
        # discard this mask / buffer, not a param
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        # Init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in
        # names and shapes
        # ignore these: just a buffer
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith(".attn.masked_bias")]
        # same, just the mask (buffer)
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # Basically the QpenAI checkpoints use a "Conv1D" module, but we only
        # want to use a vanilla Linear this means that we have to transpose
        # these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), \
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        assert config.vocab_size is not None, "config.vocab_size must be set"
        assert config.block_size is not None, "config.block_size must be set"
        self.config = config

        #: The full transformer section of the model, including the embedding
        #: and the transformer blocks
        self.transformer = nn.ModuleDict(
            {
                # The token embedding
                'wte': nn.Embedding(config.vocab_size, config.n_embd),
                # The position embedding
                'wpe': nn.Embedding(config.block_size, config.n_embd),
                # The dropout mechinism
                'drop': nn.Dropout(config.dropout),
                # The transformer blocks
                'h': nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                # The final layer normalization
                'ln_f': LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        #: The final linear projection for the logit prediction
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # With weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # Initialize all weights to standard normal
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def get_num_params(self, non_embedding: bool = True):
        """
        Return the number of parameters in the model.

        For non-embedding count (default), the position embeddings get
        subtracted.  The token embeddings would too, except due to the parameter
        sharing these params are actually used as weights in the final layer, so
        we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize our weights with random values from a gaussian distribution,
        mean of 0 and std of 0.02.  This is adapted from the original GPT-2
        implementation.

        Args:
            module: the pytorch module to initialize
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Send the input through the transformer model.  If ``targets`` is not
        ``None``, then we're in training: calculate the loss as the
        cross-entropy between the predictions and the targets.

        For ``inputs``, the input indices, we have shape ``(batch_size,
        sequence_length)``, where ``sequence_length`` is less than or equal to
        :py:attr:`config.block_size`.

        Similarly, for ``targets``, the target indices, we have shape
        ``(batch_size, sequence_length)`` with the same constraint.

        Args:
            input: The input indices.  Should have shape ``(batch_size,
                sequence_length)``.

        Keyword Arguments:
            targets: the target indices.  Should have shape ``(batch_size,
                sequence_length)``.

        Returns:
            If ``targets`` is not ``None``, return a 2-tuple containing as the
            logits of the model, and the loss.  Otherwise, return a 2-tuple
            containing as the last logit, and ``None``.

        """
        #: seq_length here is the length of our input sequences, _ is the batch size
        _, seq_length = input.size()
        assert (
            seq_length <= self.config.block_size
        ), f"Cannot forward sequence of length {seq_length}, block size is only {self.config.block_size}"
        # Construct the position indices for the input, shape (seq_length,)
        pos = torch.arange(0, seq_length, dtype=torch.long, device=input.device)

        # Forward the GPT model itself
        # ----------------------------------------
        # Transform our input token indices to embeddings of shape (batch_size,
        # seq_length, n_embd)
        tok_emb = self.transformer.wte(input)
        # Position embeddings of shape (seq_length, n_embd)
        pos_emb = self.transformer.wpe(pos)
        # Do any dropout
        x = self.transformer.drop(tok_emb + pos_emb)
        # Send the input through the transformer blocks in series
        for head in self.transformer.h:
            x = head(x)
        # apply a final layer normalization
        output = self.transformer.ln_f(x)

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            logits = self.lm_head(output)
            loss: Optional[torch.Tensor] = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # Inference-time mini-optimization: only forward the lm_head on the
            # very last position.  Note: using list [-1] to preserve the time
            # dim.
            logits = self.lm_head(output[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int) -> None:
        """
        Model surgery to decrease the block size if necessary.  For example, we
        may load the GPT2 pretrained model checkpoint (block size 1024) but want
        to use a smaller block size for some smaller, simpler model

        Args:
            block_size: The new block size to use.
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.position_embeddings.weight = nn.Parameter(
            self.transformer.position_embeddings.weight[:block_size]  # type: ignore
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def estimate_mfu(
        self,
        fwdbwd_per_iter: int,
        dt: float
    ) -> float:
        """
        Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.

        Note that this is specifc to the A100 GPU, and is not a general metric.

        Args:
            fwdbwd_per_iter: The number of forward/backward passes per iteration.
            dt: The time per iteration.
        """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu


class GPTModelLoader:

    def new(
        self,
        settings: InferenceSettings,
        init_from: str = 'scratch',
        checkpoint: Optional[Any] = None,
        vocab_size: Optional[int] = None
    ) -> GPT:
        """
        Return a new GPT model.

        Args:
            settings: The training settings
            init_from: The initialization method to use.  One of ``scratch``,
                ``resume``, or ``gpt2*``.

        Keyword Args:
            checkpoint: a checkpoint to load from, required if ``init_from`` is
                ``resume``.

        Raises:
            ValueError: the ``init_from`` setting is not recognized

        Returns:
            A GPT model
        """
        self.settings = settings
        self.checkpoint = checkpoint
        self.vocab_size = vocab_size
        self.init_from = init_from
        if init_from == 'scratch':
            model = self.from_scratch()
        elif init_from == 'resume':
            model = self.from_checkpoint()
        elif init_from.startswith('gpt2'):
            model = self.from_pretrained()
        else:
            raise ValueError(f"Unknown init_from: {init_from}")

        # crop down the model block size if desired, using model surgery
        if self.settings.block_size < model.config.block_size:
            model.crop_block_size(self.settings.block_size)
            model.config.block_size = self.settings.block_size
        # send the model to the device
        model.to(self.settings.device)
        return model

    def from_scratch(self) -> GPT:
        """
        Construct a new GPT model from scratch.

        Returns:
            A GPT model with randomly initialized parameters.
        """
        # Create a new model
        config = GPTConfig(
            n_layer=self.settings.n_layer,
            n_head=self.settings.n_head,
            n_embd=self.settings.n_embd,
            block_size=self.settings.block_size,
            dropout=self.settings.dropout,
            bias=self.settings.bias,
            vocab_size=cast(int, self.vocab_size),
        )
        return GPT(config)

    def from_checkpoint(self) -> GPT:
        """
        Construct a new GPT model from a saved checkpoint.

        Returns:
            A GPT model with parameters loaded from the checkpoint.
        """
        return GPT.from_checkpoint(self.checkpoint)

    def from_pretrained(self) -> GPT:
        """
        Construct a new GPT model from a pretrained model.

        Returns:
            A GPT model with parameters loaded from the pretrained model.
        """
        return GPT.from_pretrained(self.init_from, {'dropout': 0.0})
