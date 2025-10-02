# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

uv run tests/unit/prepare_unit_test_assets.py

cd /opt/rlkit
uv run --no-sync bash -x ./tests/run_unit.sh unit/ --ignore=unit/models/generation/ --ignore=unit/models/policy/ --cov=rlkit --cov-report=term-missing --cov-report=json --hf-gated
