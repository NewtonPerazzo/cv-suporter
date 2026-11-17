#!/bin/sh
set -eu
export OLLAMA_HOST=0.0.0.0:11434
ollama serve &
server_pid=$!
trap 'kill "$server_pid"; wait "$server_pid"' TERM INT EXIT
until OLLAMA_HOST=127.0.0.1:11434 ollama list >/dev/null 2>&1; do
    kill -0 "$server_pid"
    sleep 1
done
OLLAMA_HOST=127.0.0.1:11434 ollama pull "${EMBEDDING_MODEL:-nomic-embed-text}"
OLLAMA_HOST=127.0.0.1:11434 ollama pull "${CHAT_MODEL:-qwen3:1.7b}"
wait "$server_pid"
