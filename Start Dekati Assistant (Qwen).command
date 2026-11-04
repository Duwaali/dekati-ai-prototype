#!/bin/zsh
# Double-click to start the Dekati assistant with the Qwen 2.5 3B model instead of Llama 3.2 3B.

cd "$(dirname "$0")"
DEKATI_MODEL="qwen2.5:3b" exec "./Start Dekati Assistant.command"
