from pathlib import Path
from typing import Any, Dict

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as f:
        return tomllib.load(f)


def get_config_section(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    if section not in config:
        raise KeyError(f"Missing [{section}] section in config.")
    return dict(config[section])
