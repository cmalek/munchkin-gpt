
# munchkin-gpt

This is a reimplementation of Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT).  I reimplmented it to understand
what he was doing and to make the code more understandable to me as a long-time
professional software engineer.

It's a very small GPT model based on GPT-2 that is trainable in a reasonable
amount of time on commodity hardware with a GPU.  If you have an ARM based Mac,
this does include `mps`, the Apple ARM GPU.

## Usage

You'll need this to install all the dependencies.  I'm assuming here you have
[pyenv](https://github.com/pyenv/pyenv) installed, and Python 3.11.3 installed
via pyenv.  If not, set up your virtual environment however you want, but use
some version of python 3.11.x.

### pyenv

If you don't have pyenv and you want it, on Macs you can get it via
[homebrew](https://brew.sh/), the de-facto package manager for macOS.

```bash
brew install pyenv
pyenv install 3.11.3
```

### Install dependencies

```bash
pyenv virtualenv 3.11.3 munchkin-gpt
pyenv local munchkin-gpt
pip install --upgrade pip wheel
pip install -r requirements.txt
```

## Training a model

Now let's train a model.

```bash
$ munchkin model list-configs

Available model configs
==========================

shakespeare
openwebtext
shakespeare-char
```

There are three model definitions here.

* `shakepeare` trains a model on the text of all of Shakespeare's plays tokenized with the
  [tiktoken](https://github.com/openai/tiktoken) tokenizer from OpenAI, which
  has a pre-built vocabulary of around 50,000 tokens.  The model trained will
  have a context window of 1024 tokens, 12 layers with 12 self-attention heads
  each, and a 768 dimensional embedding space.  These are the same parameters as GPT-2.
* `openwebtext` trains a model with all of the text from the
  [openwebtext](https://skylion007.github.io/OpenWebTextCorpus/) corpus, tokenized again
  with `tiktoken`, and with the same GPT-2 parameters as `shakespeare`.
* `shakespeare-char` trains a model on all the text of Shakespeare's plays tokenized by
   character, ending up with 65 tokens or so.  The model trained will have
   a context window of 256 tokens, 6 layers with 6 self-attention heads each, and an
   embedding dimensionality of 384.  We'll train the model on 5000 iterations of the
   training data set.

The configurations for the models are in `etc/training`, and you can tweak them
if you want.  The explanations for what each setting is, and a list of all
available settings can be found in `munchkin/settings.py`.   In that file, look at
`InferenceSettings` and `TrainingSettings`, and capitalize the names of anything
you want to set in the `etc/training` `.env` file.

One thing you may want to set specifically is the `DEVICE`:

* On ARM Macs, set this to `mps`
* If you have a CUDA capable device (e.g. an nVidia graphics card) on your
  computer, set it to `cuda`
* Otherwise, you can set it to `cpu`

## shakespeare-char

Realistically, training either `shakespeare` or `openwebtext` on your commodity
computer will take days upon days.  Andrej Karpathy said that the `openwebtext`
training took around 4 days distributed across 8 nVidia A100s, which is way, way
more powerful than anything us mortals have access to.

`shakespeare-char` is much more manageable and can be trained on commodity
hardware in about few hours or so, so we'll train that one.

### Dataset tokenization and splitting

We ship with the Shakespeare dataset pre-downloaded.  It needs to be tokenized
and split into training and validation datasets.  We do this with `munckin data prepare`.

```bash
$ munchkin data prepare shakepeare_char

Processing dataset "shakespeare" with these settings:
{'dataset': 'shakespeare', 'dataset_loader_kwargs': {}, 'val_size': 0.1, 'dataset_dtype': 'uint16'}
```

The first time you run `munchkin` you'll see a delay while `pytorch` does some
pre-processing of its internals.  But after that, the delay should be minimal.

Once this finishes, you should see these files:

```
etc/shakespeare/train.bin
etc/shakespear/val.bin
```

### Training shakespeare-char

```bash

$ munchkin model train shakespeare-char

Training settings:
=================
dataset                       : shakespeare_char
dataset_loader_kwargs         : None
val_size                      : 0.1
dataset_dtype                 : uint16
n_layer                       : 6
n_head                        : 6
n_embd                        : 384
dropout                       : 0.2
bias                          : False
block_size                    : 256
device                        : mps
dtype                         : float16
name                          : shakespeare-char
eval_interval                 : 250
log_interval                  : 10
eval_iters                    : 200
eval_only                     : False
gradient_accumulation_steps   : 1
batch_size                    : 64
seed                          : 1337
always_save_checkpoint        : False
init_from                     : scratch
wandb_log                     : False
wandb_project                 : munchkin-shakespeare-char
wandb_run_name                : munchkin-shakespeare-char
learning_rate                 : 0.001
max_iters                     : 5000
weight_decay                  : 0.1
beta1                         : 0.9
beta2                         : 0.99
grad_clip                     : 1.0
decay_lr                      : True
warmup_iters                  : 100
lr_decay_iters                : 5000
min_lr                        : 0.0001
ddp                           : False
backend                       : nccl
ddp_rank                      : 0
ddp_local_rank                : 0
ddp_world_size                : 1
compile                       : False

Training Setup:
==============
Device                        : mps
Parameter data type           : float16
Tokens per iteration          : 16384
Random seed                   : 1337
Using DDP                     : False
Checkpoint file               : /Users/cmalek/src/workspace/munchkin-gpt/models/shakespeare-char/checkpoint.pt
GPT parameters                : 10646784

Optimizer:
=========
Decayed parameters            : 26, with 10,740,096 parameters
Non-decayed parameters        : 13, with 4,992 parameters
Betas                         : (0.9, 0.99)
Using fused AdamW             : False


Training
========
Estimating train loss: 100%|███████████████████████████████████████| 200/200 [00:37<00:00,  5.28it/s]
Estimating val loss: 100%|█████████████████████████████████████████| 200/200 [00:35<00:00,  5.58it/s]
Eval 0: train loss: 4.2862, val loss: 4.2804
...
```

This will go on for a while -- maybe few hours.

Every `LOG_INTERVAL` iterations (default is 10 for `shakespeare-char`) of
training through the entire dataset,  you'll see logs being written to your
terminal like so:

```
Iter 1580: loss=1.3255, elapsed=1250.08 per_iter=0.47s, mfu 0.76%  lr=8.12e-04
```

This names the iteration number, the value of the loss function (`loss`), total
elapsed time since the start of training in seconds (`elapsed`), the how long
it's taking for each iteration in seconds (`per_iter`), model FLOPS (floating
point operations per second) as a percentage of peak FLOPS for an nVidia A100
when using bfloat16 parameters (`mfu`), and the current value of the learning
rate `lr`).

As you watch the training, you should see the `loss` generally decrease towards
zero, and the `lr` decay (this is normal) and hopefully your `per_iter` stays
constant.

### Periodic model validation and parameter saving

Every `EVAL_INTERVAL` iterations (250 by default for `shakespeare-char`), the
training algorithm will evaluate the performance of the model against both the
training data and the validation data sets.

If the validation loss is less than it was the last time we checked, we save our
model parameters to a checkpoint file so we can stop training and pick up where
we left off.  This checkpoint file will also be used during the model text
generation later.  You'll know this happened because we logged it and the report
of training loss vs validation loss will be green.

If you see a few validation tests that log in red and don't produce a checkpoint
file, you can stop the training -- the model is becoming overfit and won't get better.

When I train `shakespeare-char`, I bottom out generally at a valiation loss of
1.5 or so.


### Generate some text!

We use `munchkin model generate` to generate text from our saved model
parameters.  You can either provide a prompt or just let it generate from
nothing.  Do `munchkin model generate --help` to see what you can tweak about
the generation process.

One important thing to note is that the model generates *all* the text before printing
anything, so if you use a high value of ``--max-tokens``, it may take a while
before you see any output.

```
munchkin model generate --max-tokens 256 shakespeare-char

Generation Setup:
================
Device                        : mps
Random seed                   : 1337
Temperature                   : 0.8
Top k                         : None
Compile                       : False

Model configuration:
===================
block_size                    : 256
vocab_size                    : 65
n_layer                       : 6
n_head                        : 6
n_embd                        : 384
dropout                       : 0.0
bias                          : False
Loading state dict ... Done

Forst Senator:
He has leisure from me arms.

CORIOLANUS:
The people's common his general.

First Servingman:
The means are done for their liberty.

First Citizen:
Good friends, or else your mother.

CORIOLANUS:
Our grace I die them, you that ha'
Woe togeth
------------------------------------------------------------
```

Not great, right?  But it does work and shows that our model was indeed trained,
and that character based tokenization and training is pretty terrible if you want
something that sounds like Shakespeare!
