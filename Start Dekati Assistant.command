#!/bin/zsh
# Double-click to start the Dekati assistant and open it in the browser. Close this window to stop it.

cd "$(dirname "$0")"

PY="${DEKATI_PYTHON:-$HOME/opt/anaconda3/envs/dekati-ai/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"
URL="http://127.0.0.1:8000"

echo "Starting the Dekati assistant..."

if ! curl -s -m 2 http://localhost:11434/api/tags > /dev/null; then
  echo "Starting Ollama..."
  open -a Ollama
  for i in {1..30}; do
    curl -s -m 2 http://localhost:11434/api/tags > /dev/null && break
    sleep 2
  done
fi

MODEL_ARGS=()
[[ -n "$DEKATI_MODEL" ]] && MODEL_ARGS=(--model "$DEKATI_MODEL") && echo "Language model: $DEKATI_MODEL"

HF_HUB_OFFLINE=1 "$PY" src/ui/app.py --port 8000 "${MODEL_ARGS[@]}" &
SERVER=$!
trap "kill $SERVER 2>/dev/null" EXIT

echo "Loading the search index (this can take a minute)..."
for i in {1..120}; do
  curl -s -m 2 "$URL" > /dev/null && break
  sleep 2
done

open "$URL"
echo ""
echo "The assistant is open in your browser: $URL"
echo "Test Bench: $URL/test"
echo "Keep this window open. Close it to stop the assistant."
wait $SERVER
