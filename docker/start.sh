#!/usr/bin/env bash
set -e

# If JUPYTER_TOKEN env var is empty, Jupyter will start without a token (insecure).
# Recommended: always set JUPYTER_TOKEN in Runpod's env settings.
: "${JUPYTER_TOKEN:=}"

echo "Starting JupyterLab..."
if [ -z "$JUPYTER_TOKEN" ]; then
  echo "WARNING: JUPYTER_TOKEN is not set — starting without token (INSECURE)."
fi

exec jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --allow-root \
  --ServerApp.token="${JUPYTER_TOKEN}" \
  --ServerApp.allow_origin='*' \
  --ServerApp.root_dir=/workspace
