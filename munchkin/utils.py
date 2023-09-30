from typing import Any, Dict, Optional

import click


def pprint_dict(
    d: Dict[str, Any],
    indent: int = 0,
    align: Optional[int] = None,
    title: Optional[str] = None,
) -> None:
    """
    Pretty print a dictionary.

    Args:
        d: The dictionary to print
        indent: The number of spaces to indent

    Keyword Args:
        align: The number of spaces to align the keys to
        title: The title to print before the dictionary
    """
    if title is not None:
        click.secho(f'\n{title}:', fg='green')
        click.secho('=' * len(title), fg='green')
    if not d:
        click.secho('None', fg='white')
        return
    if align is None:
        align = max(len(str(k)) for k in d)
    for k, v in d.items():
        click.secho(f'{" " * indent}{k.ljust(align)}: ', fg='cyan', nl=False)
        if isinstance(v, dict):
            pprint_dict(v, indent + 2)
        else:
            click.secho(f'{v}', fg='white')
