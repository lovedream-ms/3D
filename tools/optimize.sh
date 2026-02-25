#!/bin/bash

# ==============================================
#  脚本模板                              配置部分
# ==============================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
OS_VERSION="$(uname -r)"
MACHINE="$(uname -m)"
PROFILE_LOG_DIR="${PROFILE_LOG_DIR:-${PROJECT_ROOT}/results/machine/profile}"

# ==============================================
# 颜色定义部分
# ==============================================
RESET="\033[0m"
HIGH="\033[1;34m" # 需要手动使用RESET
ERROR="\033[1;31m[ERROR]: $RESET"
WARNING="\033[1;33m[WARNING]: $RESET"
INFO="\033[1;32m[INFO]: $RESET"
SUCCESS="\033[1;34m[SUCCESS]: $RESET"
NOTICE="\033[1;34m[NOTICE]: $RESET"

# ==============================================
# 函数定义部分
# ==============================================
# 错误处理函数
on_error() {
    local lineno="$1"
    local cmd="$2"
    local code="$3"

    echo -e "${ERROR}${HIGH}Command \"${cmd}\" Failed${RESET} at ${HIGH}Line ${lineno}${RESET} with Exit Code ${HIGH}${code}${RESET}"
    echo -e "${ERROR}${HIGH}Call Stack:${RESET}"
    for ((i = ${#FUNCNAME[@]} - 1; i >= 1; i--)); do
        echo -e "  in ${HIGH}${FUNCNAME[i]}()${RESET} at ${BASH_SOURCE[i]}:${HIGH}${BASH_LINENO[i - 1]}${RESET}"
    done
    exit "$code"
}
trap 'on_error $LINENO "$BASH_COMMAND" $?' ERR
set -eE
set -o errtrace

# ==============================================
# 主程序部分
# ==============================================
main() {
    echo -e "${INFO}Running on ${HIGH}${OS_VERSION}${RESET} (${HIGH}${MACHINE}${RESET})"
    # uv run viztracer -o results/human/timeline.html --open --log_gc src/Main.py
    # uv run --env-file .env tools/optimize.py
    PYTHONPATH=src uv run tools/optimize.py
    echo -e "${SUCCESS}Script executed successfully!${RESET}"
}

main
