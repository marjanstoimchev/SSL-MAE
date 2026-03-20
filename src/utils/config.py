import os
import sys
from omegaconf import OmegaConf


def load_config(config_path, cli_args=None):
    """Load YAML config with defaults resolution and CLI overrides.

    Supports a `defaults` key in the YAML file to inherit from a base config
    in the same directory. Example:
        defaults:
          - default       # loads default.yaml from same dir

    Args:
        config_path: Path to the YAML config file.
        cli_args: List of CLI override strings (e.g., ["training.epochs=50"]).
                  If None, parses from sys.argv.

    Returns:
        OmegaConf DictConfig with dot-notation access.
    """
    cfg = OmegaConf.load(config_path)
    config_dir = os.path.dirname(os.path.abspath(config_path))

    # Resolve defaults (base configs to inherit from)
    if "defaults" in cfg:
        defaults = cfg.pop("defaults")
        base_cfg = OmegaConf.create({})
        for default in defaults:
            default_path = os.path.join(config_dir, f"{default}.yaml")
            base_cfg = OmegaConf.merge(base_cfg, OmegaConf.load(default_path))
        cfg = OmegaConf.merge(base_cfg, cfg)

    # Apply CLI overrides
    if cli_args is None:
        cli_args = [arg for arg in sys.argv[1:] if '=' in arg]

    if cli_args:
        cli_cfg = OmegaConf.from_dotlist(cli_args)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg


def parse_args():
    """Parse --config argument and return (config_path, remaining_overrides)."""
    args = sys.argv[1:]
    config_path = None
    overrides = []

    i = 0
    while i < len(args):
        if args[i] == '--config' and i + 1 < len(args):
            config_path = args[i + 1]
            i += 2
        elif '=' in args[i]:
            overrides.append(args[i])
            i += 1
        else:
            i += 1

    if config_path is None:
        raise ValueError("Must provide --config <path_to_yaml>")

    return config_path, overrides
