class ProviderUnavailable(Exception):
    """Safe marker. Never include external response bodies in error messages."""


class InvalidMaterial(ValueError):
    pass
