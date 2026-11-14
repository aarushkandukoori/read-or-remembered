#!/bin/zsh
# Saves a DeepSeek API key to ~/Projects/read-or-remembered/.env (only you can read it).
# The key is typed hidden and never shown on screen or in chat.
read -rs "k?Paste your DeepSeek API key (hidden), then press Enter: "
echo
if [[ -z "$k" ]]; then echo "Nothing pasted. Run this again."; exit 1; fi
umask 077
printf 'DEEPSEEK_API_KEY=%s\n' "$k" > ~/Projects/read-or-remembered/.env
unset k
echo "Saved. You can close this tab."
