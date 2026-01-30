"""LLM Cost Predictor - CLI entry point.

Monte Carlo simulation for multi-model LLM cost prediction.
"""

import click


@click.group()
@click.option('--config-dir', default='config', help='Configuration directory')
@click.pass_context
def cli(ctx, config_dir: str) -> None:
    """LLM Cost Predictor - Monte Carlo simulation for multi-model costs."""
    ctx.ensure_object(dict)
    ctx.obj['config_dir'] = config_dir


if __name__ == '__main__':
    cli()
