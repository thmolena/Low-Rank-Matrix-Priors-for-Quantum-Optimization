"""Deterministic structured-operator audit package."""

__version__ = "1.1.0"


def reproduce():
    """Run the registered experiment through a lazy import."""
    from .experiment import reproduce as run

    return run()

__all__ = ["__version__", "reproduce"]
