from pathlib import Path

import yaml

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "policies"


def policy(settings, name):
    """Читает YAML-политику из policy_dir либо встроенного каталога; ошибки чтения не подавляет."""
    directory = Path(settings.policy_dir) if settings.policy_dir else DEFAULT_DIR
    return yaml.safe_load((directory / name).read_text())
