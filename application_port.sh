#!/bin/bash

DEFAULT_APPLICATION_PORT=8080
APPLICATION_PORT_FILE="${APPLICATION_PORT_FILE:-/u01/aipoc/props/application_port.txt}"

resolve_application_port() {
  local configured_port="${APPLICATION_PORT:-}"
  local port_number

  if [ -z "$configured_port" ] && [ -r "$APPLICATION_PORT_FILE" ]; then
    IFS= read -r configured_port < "$APPLICATION_PORT_FILE" || true
  fi

  configured_port="${configured_port:-$DEFAULT_APPLICATION_PORT}"

  if [[ ! "$configured_port" =~ ^[0-9]{1,5}$ ]]; then
    echo "APPLICATION_PORT must be an integer between 1 and 65535." >&2
    return 1
  fi

  port_number=$((10#$configured_port))
  if ((port_number < 1 || port_number > 65535)); then
    echo "APPLICATION_PORT must be an integer between 1 and 65535." >&2
    return 1
  fi

  case "$port_number" in
    3000 | 5432 | 7932)
      echo "APPLICATION_PORT $port_number is reserved by another No.1-RAG service." >&2
      return 1
      ;;
  esac

  APPLICATION_PORT="$port_number"
  export APPLICATION_PORT
}

resolve_application_port
