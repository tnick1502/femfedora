from cryptography.fernet import Fernet, InvalidToken
from pydantic import SecretStr


class SecretConfigurationError(ValueError):
    pass


def reveal(value: str | SecretStr) -> str:
    """Извлекает строку из SecretStr; обычную строку возвращает без преобразования."""
    return value.get_secret_value() if isinstance(value, SecretStr) else value


def encrypt_key(value: str, master_key: str) -> str:
    """Шифрует API-ключ Fernet-ключом master_key и возвращает строку шифротекста.

    Пустой или некорректный ключ шифрования вызывает SecretConfigurationError.
    Исходный API-ключ в сообщение исключения не включается.
    """
    if not master_key:
        raise SecretConfigurationError(
            "Задайте MODEL_SECRETS_KEY в .env для ввода API-ключей через панель"
        )
    try:
        return Fernet(master_key.encode()).encrypt(value.encode()).decode()
    except (ValueError, TypeError):
        raise SecretConfigurationError("Некорректный MODEL_SECRETS_KEY") from None


def decrypt_key(value: str, master_key: str) -> str:
    """Расшифровывает сохранённый API-ключ с помощью master_key.

    При неверном ключе, повреждённом шифротексте или ошибке проверки подписи
    вызывает SecretConfigurationError с фиксированным сообщением.
    """
    try:
        return Fernet(master_key.encode()).decrypt(value.encode()).decode()
    except (ValueError, TypeError, InvalidToken):
        raise SecretConfigurationError(
            "Не удалось расшифровать API-ключ; проверьте MODEL_SECRETS_KEY"
        ) from None
