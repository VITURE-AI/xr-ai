# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The built-in ``vlm`` procedure backend."""

from .backend import VlmBackend, create_backend
from .config import VlmBackendConfig

__all__ = ["VlmBackend", "VlmBackendConfig", "create_backend"]
