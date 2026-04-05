#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HUGO_DIR="${HUGO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
AI_PROXY_ENV_FILE="${AI_PROXY_ENV_FILE:-}"
AI_PROXY_SHELL_RC="${AI_PROXY_SHELL_RC:-}"

if [[ -n "$AI_PROXY_ENV_FILE" && -f "$AI_PROXY_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$AI_PROXY_ENV_FILE"
  set +a
fi

if [[ -n "$AI_PROXY_SHELL_RC" && -f "$AI_PROXY_SHELL_RC" ]]; then
  if [[ "$AI_PROXY_SHELL_RC" == *.zshrc ]] && command -v zsh >/dev/null 2>&1; then
    exec zsh -lc "source \"$AI_PROXY_SHELL_RC\" >/dev/null 2>&1 || true; cd \"$HUGO_DIR\"; exec npm run ai-proxy"
  fi

  if [[ "$AI_PROXY_SHELL_RC" == *.bashrc ]] || [[ "$AI_PROXY_SHELL_RC" == *.profile ]] || [[ "$AI_PROXY_SHELL_RC" == *.bash_profile ]]; then
    exec bash -lc "source \"$AI_PROXY_SHELL_RC\" >/dev/null 2>&1 || true; cd \"$HUGO_DIR\"; exec npm run ai-proxy"
  fi

  set +u
  # shellcheck disable=SC1090
  source "$AI_PROXY_SHELL_RC" >/dev/null 2>&1 || true
  set -u
fi

cd "$HUGO_DIR"
exec npm run ai-proxy
