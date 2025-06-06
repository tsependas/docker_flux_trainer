#!/usr/bin/env bash
set -e

exec python -u /app/download_model.py

exec python -u /app/train.py

