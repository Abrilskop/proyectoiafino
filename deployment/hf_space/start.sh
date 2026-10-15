#!/usr/bin/env bash
set -e
exec uvicorn deployment.app:app --host 0.0.0.0 --port "${PORT:-7860}"
