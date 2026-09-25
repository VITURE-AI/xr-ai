<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Applications

Build applications that are not repository samples under `apps/<your-app>/`.
Start with the
[build-your-application guide](https://nvidia.github.io/xr-ai/latest/guides/building-your-app.html).

Repository dependency, lint, SPDX, sample catalog, and test discovery exclude
application-owned files in this directory. Application owners choose the
licensing and additional validation appropriate for their code. This fork
tracks its applications, so the upstream `apps/*` ignore rule is removed from
the root `.gitignore`.
