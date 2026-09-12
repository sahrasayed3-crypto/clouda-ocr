"""Standalone local web dashboard for Clouda Lab."""

from .app import create_app
from .settings import LabSettings

__all__ = ["LabSettings", "create_app"]
