#!/bin/bash

APT_LOCK_TIMEOUT_SECONDS="${APT_LOCK_TIMEOUT_SECONDS:-300}"
APT_MAX_ATTEMPTS="${APT_MAX_ATTEMPTS:-5}"
APT_RETRY_DELAY_SECONDS="${APT_RETRY_DELAY_SECONDS:-10}"

apt_get_with_retry() {
    local attempt=1
    local exit_code=1
    local -a apt_command=(
        apt-get
        -o
        "DPkg::Lock::Timeout=${APT_LOCK_TIMEOUT_SECONDS}"
    )

    if [ "${EUID}" -ne 0 ]; then
        apt_command=(sudo "${apt_command[@]}")
    fi

    while [ "${attempt}" -le "${APT_MAX_ATTEMPTS}" ]; do
        echo "APT attempt ${attempt}/${APT_MAX_ATTEMPTS} (lock timeout: ${APT_LOCK_TIMEOUT_SECONDS}s): ${apt_command[*]} $*"

        if "${apt_command[@]}" "$@"; then
            return 0
        else
            exit_code=$?
        fi

        if [ "${attempt}" -lt "${APT_MAX_ATTEMPTS}" ]; then
            echo "APT command failed with exit code ${exit_code}. Retrying in ${APT_RETRY_DELAY_SECONDS} seconds..."
            sleep "${APT_RETRY_DELAY_SECONDS}"
        fi

        attempt=$((attempt + 1))
    done

    echo "APT command failed after ${APT_MAX_ATTEMPTS} attempts."
    return "${exit_code}"
}
