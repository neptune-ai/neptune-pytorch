import contextlib

__all__ = ["__version__"]

from importlib.metadata import (
    PackageNotFoundError,
    version,
)
from importlib.util import find_spec

if not (find_spec("neptune_scale")):
    msg = """
            The Neptune Scale client library was not found.

            Install the neptune-scale package with
                `pip install -U neptune-scale`

            Need help? -> https://docs.neptune.ai/setup"""
    raise PackageNotFoundError(msg)

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("neptune-pytorch")
