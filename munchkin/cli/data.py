from pathlib import Path

import click
from munchkin.data import Dataset
from munchkin.settings import DatasetSettings

from .cli import cli


@cli.group(
    short_help="Process datasets",
    name='data'
)
def data_group():
    pass


@data_group.command('list-configs', short_help="List known dataset configs")
def data_list_datasets() -> None:
    """
    List known dataset configs.


    Raises:
        click.ClickException: no dataset definition for the given dataset
    """
    click.secho('Available dataset configs', fg='green')
    click.secho('==========================\n', fg='green')
    for dataset in Path(__file__).parent.parent.parent.glob('etc/dataset/*'):
        print(dataset.name)


@data_group.command('process', short_help="Pre-process a dataset")
@click.argument('dataset_name')
def data_preprocess_dataset(dataset_name: str) -> None:
    """
    Pre-process a dataset.

    Args:
        dataset: The name of the dataset to pre-process

    Raises:
        click.ClickException: no dataset definition for the given dataset
    """
    config = Path(__file__).parent.parent.parent / 'etc/dataset' / dataset_name / 'config.env'
    if not config.exists():
        raise click.ClickException(f'No such file as "{config}"')
    settings = DatasetSettings(_env_file=config)
    print(f'Processing dataset "{dataset_name}" with these settings:')
    print(settings.model_dump())
    dataset = Dataset(settings)
    dataset.loader.tokenize()
