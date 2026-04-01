#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Building image..."
podman compose build

echo "Restarting aineko service..."
systemctl --user restart aineko.service

echo "Waiting for startup..."
sleep 8

echo "--- Recent logs ---"
podman logs --tail 15 aineko_aineko_1

echo ""
echo "Done."
