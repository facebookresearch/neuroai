# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import getpass
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any


def _is_interactive() -> bool:
    """Return True when stdin is attached to a terminal."""
    return hasattr(sys.stdin, "isatty") and sys.stdin.isatty()


def get_default_config_path() -> Path:
    """Get the default configuration file path."""
    return Path.home() / ".neuralbench" / "config.json"


def prompt_user_for_path(
    key: str, description: str, default_value: str | None = None
) -> str:
    """Prompt user for a path with optional default value."""
    if default_value:
        response = input(f"{description}\n[Default: {default_value}]: ").strip()
        return response if response else default_value
    else:
        response = input(f"{description}: ").strip()
        while not response:
            print("This field is required. Please provide a path.")
            response = input(f"{description}: ").strip()
        return response


def setup_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Set up neuralbench configuration.

    If config doesn't exist, prompt user for paths.

    Args:
        config_path: Path to config file. If None, uses default location.

    Returns:
        Dictionary with configuration values.
    """
    if config_path is None:
        print("\nNeuralbench Configuration Setup")
        print("=" * 50)
        response = (
            input(
                "\nConfiguration file not found.\n"
                "Do you want to use the default location (~/.neuralbench/config.json)? [Y/n]: "
            )
            .strip()
            .lower()
        )

        if response in ["n", "no"]:
            custom_path = input("Enter custom configuration file path: ").strip()
            config_path = Path(custom_path).expanduser()
        else:
            config_path = get_default_config_path()

    # If config exists, load and return it
    if config_path.exists():
        with config_path.open() as f:
            config = json.load(f)
        print(f"\nLoaded configuration from: {config_path}")
        return config

    # Config doesn't exist, prompt for values
    print(f"\nSetting up new configuration at: {config_path}")
    print("-" * 50)

    config = {}

    # Get username and entity name
    username = getpass.getuser()
    config["USER"] = username
    config["ENTITY_NAME"] = username

    # Default project name
    config["PROJECT_NAME"] = "neuralbench"

    # SLURM defaults (can be overridden later in config.json)
    config["SLURM_PARTITION"] = ""
    config["SLURM_CONSTRAINT"] = ""
    config["N_CPUS"] = 10

    # Prompt for paths
    print("\nPlease provide the following paths:")
    print()

    config["CACHE_DIR"] = prompt_user_for_path(
        "CACHE_DIR",
        "CACHE_DIR - Where to cache intermediate results from experiments",
    )

    config["SAVE_DIR"] = prompt_user_for_path(
        "SAVE_DIR",
        "SAVE_DIR - Where to save experiment results",
    )

    config["DATA_DIR"] = prompt_user_for_path(
        "DATA_DIR",
        "DATA_DIR - Where to download and store datasets",
    )

    # Create directories if they don't exist
    for key in ["CACHE_DIR", "SAVE_DIR", "DATA_DIR"]:
        path = Path(config[key])
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")

    # Prompt for W&B host
    print()
    wandb_host = input(
        "WANDB_HOST - Weights & Biases server URL\n"
        "(leave empty to skip, e.g. https://wandb.ai/): "
    ).strip()
    config["WANDB_HOST"] = wandb_host

    # Save configuration
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nConfiguration saved to: {config_path}")
    print("=" * 50)
    print()

    return config


def _default_config() -> dict[str, Any]:
    """Return a minimal config with temporary paths for non-interactive use."""
    import tempfile

    base = Path(tempfile.gettempdir()) / "neuralbench"
    username = getpass.getuser()
    config = {
        "USER": username,
        "ENTITY_NAME": username,
        "PROJECT_NAME": "neuralbench",
        "CACHE_DIR": str(base / "cache"),
        "SAVE_DIR": str(base / "save"),
        "DATA_DIR": str(base / "data"),
        "WANDB_HOST": "",
        "SLURM_PARTITION": "",
        "SLURM_CONSTRAINT": "",
        "N_CPUS": 10,
    }
    for key in ["CACHE_DIR", "SAVE_DIR", "DATA_DIR"]:
        Path(str(config[key])).mkdir(parents=True, exist_ok=True)
    return config


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load neuralbench configuration.

    The config file location is resolved in order:
    1. Explicit *config_path* argument
    2. ``NEURALBENCH_CONFIG`` environment variable
    3. Default ``~/.neuralbench/config.json``

    Args:
        config_path: Path to config file. If None, checks env var then default.

    Returns:
        Dictionary with configuration values.
    """
    if config_path is None:
        env_path = os.environ.get("NEURALBENCH_CONFIG")
        if env_path:
            config_path = Path(env_path).expanduser()
        else:
            config_path = get_default_config_path()

    if not config_path.exists():
        if not _is_interactive():
            return _default_config()
        return setup_config(config_path)

    with config_path.open() as f:
        return json.load(f)


# Global config instance (will be initialized when module is imported)
_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    """Get the current configuration, loading it if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


if TYPE_CHECKING:
    USER: str
    ENTITY_NAME: str
    PROJECT_NAME: str
    DATA_DIR: str
    CACHE_DIR: str
    SAVE_DIR: str
    WANDB_HOST: str
    SLURM_PARTITION: str
    SLURM_CONSTRAINT: str
    N_CPUS: int


def _initialize_module_vars() -> None:
    """Initialize module-level variables from config."""
    global USER, ENTITY_NAME, PROJECT_NAME, DATA_DIR, CACHE_DIR, SAVE_DIR
    global WANDB_HOST, SLURM_PARTITION, SLURM_CONSTRAINT, N_CPUS
    config = get_config()
    USER = config["USER"]
    ENTITY_NAME = config["ENTITY_NAME"]
    PROJECT_NAME = config["PROJECT_NAME"]
    DATA_DIR = config["DATA_DIR"]
    CACHE_DIR = config["CACHE_DIR"]
    SAVE_DIR = config["SAVE_DIR"]
    WANDB_HOST = config.get("WANDB_HOST", "")
    SLURM_PARTITION = config.get("SLURM_PARTITION", "")
    SLURM_CONSTRAINT = config.get("SLURM_CONSTRAINT", "")
    N_CPUS = config.get("N_CPUS", 10)


# Lazy module-level config variables for YAML compatibility
# (``!!python/name:neuralbench.config_manager.DATA_DIR`` etc.).
# Values are resolved on first access via ``__getattr__`` so that
# importing this module does not trigger file I/O or interactive prompts.
_LAZY_CONFIG_KEYS = {
    "USER",
    "ENTITY_NAME",
    "PROJECT_NAME",
    "DATA_DIR",
    "CACHE_DIR",
    "SAVE_DIR",
    "WANDB_HOST",
    "SLURM_PARTITION",
    "SLURM_CONSTRAINT",
    "N_CPUS",
}
_initialized = False


def _ensure_initialized() -> None:
    global _initialized
    if not _initialized:
        _initialize_module_vars()
        _initialized = True


def __getattr__(name: str) -> Any:
    if name in _LAZY_CONFIG_KEYS:
        _ensure_initialized()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
