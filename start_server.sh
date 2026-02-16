#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

uv run uvicorn paper_analysis_deepagents.api:app --host 0.0.0.0 --port 8000 --reload
