#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to publish site output."
  exit 1
fi

cd "$ROOT_DIR"

npm run build

rm -rf "$ROOT_DIR/_astro"
cp -r "$ROOT_DIR/dist/_astro" "$ROOT_DIR/_astro"
cp "$ROOT_DIR/dist/index.html" "$ROOT_DIR/index.html"
cp "$ROOT_DIR/dist/docs.html" "$ROOT_DIR/docs.html"
cp "$ROOT_DIR/dist/hardware.html" "$ROOT_DIR/hardware.html"
cp "$ROOT_DIR/dist/demos.html" "$ROOT_DIR/demos.html"
cp "$ROOT_DIR/dist/community.html" "$ROOT_DIR/community.html"

echo "Published Astro output to repository root."
