#!/usr/bin/env bash
# Build the Wasm telemetry agent and drop it next to the live demo.
#
#   ./agent/build.sh
#
# Needs the wasm target once:  rustup target add wasm32-unknown-unknown
set -euo pipefail
cd "$(dirname "$0")"

cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/cit_agent.wasm ../web/agent.wasm

size=$(wc -c < ../web/agent.wasm)
echo "built ../web/agent.wasm  (${size} bytes)"
