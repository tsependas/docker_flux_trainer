#!/usr/bin/env bash
set -e

python /app/download_model.py

runpodctl stop pod $RUNPOD_POD_ID
