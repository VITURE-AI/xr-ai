# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible speech-to-text and text-to-speech over hosted DashScope models."""

from .app import create_app
from .config import ConfigError, Settings, load_settings, settings_from_dict

__all__ = ["ConfigError", "Settings", "create_app", "load_settings", "settings_from_dict"]
