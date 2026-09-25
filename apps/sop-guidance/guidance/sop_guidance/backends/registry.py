# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a procedure's ``backend:`` name to a backend factory.

A backend is found through the ``sop_guidance.backends`` entry-point group, so
a new backend is a separately installed package and the host does not change.
A ``module:attribute`` spelling is accepted too, for a backend that lives in
the application tree without its own distribution.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points

from .base import BackendFactory

ENTRY_POINT_GROUP = "sop_guidance.backends"


class UnknownBackendError(LookupError):
    """No installed backend matches a procedure's ``backend:`` value."""


def available_backends() -> list[str]:
    """Names registered in the entry-point group, sorted."""

    return sorted(ep.name for ep in entry_points(group=ENTRY_POINT_GROUP))


def resolve_backend(name: str) -> BackendFactory:
    """Return the factory for *name*.

    Raises :class:`UnknownBackendError` naming the installed backends when
    nothing matches.
    """

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == name:
            return ep.load()
    if ":" in name:
        module_name, _, attribute = name.partition(":")
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attribute)
        except (ImportError, AttributeError) as exc:
            raise UnknownBackendError(f"backend {name!r} cannot be imported: {exc}") from exc
    raise UnknownBackendError(
        f"unknown backend {name!r}; installed: {', '.join(available_backends()) or 'none'}"
    )


__all__ = ["ENTRY_POINT_GROUP", "UnknownBackendError", "available_backends", "resolve_backend"]
