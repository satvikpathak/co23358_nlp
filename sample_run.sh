#!/usr/bin/env bash
# Example curl to test the API (ensure backend is running at localhost:8000)
set -euo pipefail

curl -sS -X POST "http://127.0.0.1:8000/rewrite" \
  -H "Content-Type: application/json" \
  -d '{"text":"Why did you miss the deadline?", "tone":"polite"}' | jq
