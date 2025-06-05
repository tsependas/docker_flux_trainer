#!/usr/bin/env bash
set -e

python /app/download_model.py

python /app/train.py

