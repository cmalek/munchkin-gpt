"""
This file is used as the entrypoint for the command line interface.
"""


def main() -> None:
    from munchkin.cli import cli  #
    cli(obj={})


if __name__ == '__main__':
    main()
