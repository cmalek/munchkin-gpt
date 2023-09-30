from pathlib import Path
from pprint import pprint
from typing import Optional

import click
from munchkin.train import GPTTrainer, GPTEvaluator
from munchkin.settings import TrainingSettings
from munchkin.generate import GPTGenerator

from ..utils import pprint_dict
from .cli import cli


@cli.group(
    short_help="Use models",
    name='model'
)
def model_group():
    pass


@model_group.command('list-configs', short_help="List available configs")
def training_list() -> None:
    """
    List available model configs.
    """
    click.secho('Available model configs', fg='green')
    click.secho('==========================\n', fg='green')
    for config in Path(__file__).parent.parent.parent.glob('etc/training/*.env'):
        name = str(config.name).replace('.env', '')
        print(f'{name}')


@model_group.command('train', short_help="Train a model")
@click.argument('config_file')
def model_train(config_file: str) -> None:
    """
    Train a model.

    Args:
        config: The name of the training configuration to use

    Raises:
        click.ClickException: no training configuration found for the given name
    """
    config = Path(__file__).parent.parent.parent / 'etc/training' / f'{config_file}.env'
    if not config.exists():
        raise click.ClickException(f'No such file as "{config}"')
    settings = TrainingSettings(_env_file=config)
    pprint_dict(settings.model_dump(), title='Training settings', align=30)
    trainer = GPTTrainer(settings)
    trainer.train()


@model_group.command('evaulate', short_help="Evaluate a model")
@click.argument('config_file')
def model_evaluate(config_file: str) -> None:
    """
    Evaulate a model, printing the training and validation set losses.

    Args:
        config: The name of the model configuration to use

    Raises:
        click.ClickException: no configuration found for the given name
        click.ClickException: no checkpoint found for the given name
    """
    config = Path(__file__).parent.parent.parent / 'etc/training' / f'{config_file}.env'
    if not config.exists():
        raise click.ClickException(f'No such file as "{config}"')
    checkpoint_path = Path(__file__).parent.parent.parent / 'models' / config_file / 'checkpoint.pt'
    if not checkpoint_path.exists():
        raise click.ClickException(f'No checkpoint found for "{config_file}"')
    settings = TrainingSettings(_env_file=config)
    settings.init_from = 'resume'
    settings.eval_only = True

    evaluator = GPTEvaluator(settings)
    losses = evaluator.estimate_loss(progress=True)
    click.secho(
        f"\nEvaluation: train loss: {losses['train']:.4f}, "
        f"val loss: {losses['val']:.4f}",
    )


@model_group.command('generate', short_help="Make a model generate text")
@click.option(
    '--prompt',
    default="\n",
    help="The prompt to use for generation",
)
@click.option(
    '--prompt-file',
    default=None,
    type=click.Path(exists=True),
    help="A file that contains the prompt to use for generation",
)
@click.option(
    '--temperature',
    '-t',
    default=0.8,
    type=float,
    help="The temperature to use for generation",
)
@click.option(
    '--top-k',
    '-k',
    type=int,
    default=None,
    help="Choose a token randomly from the top k tokens",
)
@click.option(
    '--random-seed',
    default=1337,
    help="Choose a token randomly from the top k tokens",
)
@click.option(
    '--iterations',
    default=1,
    help="Do this many iterations of generation",
)
@click.option(
    '--max-tokens',
    default=50,
    help="Maximum number of tokens to generate per iteration",
)
@click.argument('model_name')
def model_generate(
    prompt: str,
    prompt_file: Optional[str],
    temperature: float,
    top_k: int,
    random_seed: int,
    iterations: int,
    max_tokens: int,
    model_name: str
) -> None:
    """
    Make a model generate text.

    If both --prompt and --prompt-file are specified, --prompt-file will be
    used.

    Args:
        config: The name of the model configuration to use

    Raises:
        click.ClickException: no configuration found for the given name
        click.ClickException: no checkpoint found for the given name
    """
    checkpoint_path = Path(__file__).parent.parent.parent / 'models' / model_name / 'checkpoint.pt'
    if not checkpoint_path.exists():
        raise click.ClickException(f'No checkpoint found for "{model_name}"')
    if prompt_file is not None:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read()
    generator = GPTGenerator(
        model_name,
        device='mps',
        random_seed=random_seed,
        temperature=temperature,
        top_k=top_k,
    )
    for _ in range(iterations):
        click.secho(generator.generate(prompt, max_tokens))
        click.secho('-' * 60, fg='green')
