import pytest
from pydantic import ValidationError

from emo_core.core.config import Settings


@pytest.mark.parametrize(
    "kwargs",
    [
        {"app_env": "production", "demo_auth_enabled": True},
        {"app_env": "production", "ai_mode": "mock"},
        {"ai_mode": "live"},
        {"jev_model": "jev-latest"},
    ],
)
def test_invalid_config_fails_loudly(kwargs):
    """Проверяет явный отказ при некорректной конфигурации вместо скрытых значений по умолчанию."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **kwargs)


def test_production_has_no_dev_endpoints(tmp_path):
    """Проверяет отсутствие dev-маршрутов в приложении production."""
    import shutil

    import yaml

    from emo_core.main import create_app
    from emo_core.services.policies import DEFAULT_DIR

    with pytest.raises(ValueError, match="documented approval"):
        create_app(
            Settings(_env_file=None, app_env="production", jwt_jwks_url="https://id.test/jwks")
        )
    shutil.copytree(DEFAULT_DIR, tmp_path / "policies")
    path = tmp_path / "policies/safety_rules.yaml"
    data = yaml.safe_load(path.read_text())
    data.update(
        clinical_review_status="approved",
        reviewed_by="synthetic-test-fixture",
        approval_reference="not-a-real-approval",
    )
    path.write_text(yaml.safe_dump(data))
    app = create_app(
        Settings(
            _env_file=None,
            app_env="production",
            jwt_jwks_url="https://id.test/jwks",
            policy_dir=str(tmp_path / "policies"),
        )
    )
    assert not any(getattr(r, "path", "").startswith("/dev/") for r in app.routes)
