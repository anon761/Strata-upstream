#!/bin/sh
# Strata fork installer for a bare Debian 12 or 13 (amd64): installs every dependency (system packages, NVIDIA
# driver and CUDA toolkit, the Python environment), builds the engine for this machine's GPUs and, with
# --gguf FILE, proposes settings for that model on this hardware and prepares it after you confirm.
#
#   sudo ./install.sh                         install
#   sudo ./install.sh --gguf /models/m.gguf   install and prepare a model
#   ./install.sh --help                       all options
#
# No environment variables are needed. The work is done by tools/install/strata_install.py, which this script
# starts with the system's Python once that is there.
set -eu
cd "$(dirname "$0")"

for arg in "$@"; do
  case "$arg" in
    -h|--help) exec python3 tools/install/strata_install.py --help ;;
  esac
done

if [ ! -r /etc/os-release ]; then
  echo "This installer supports Debian 12 and 13 (no /etc/os-release here)." >&2
  exit 1
fi
# shellcheck source=/dev/null
. /etc/os-release
if [ "${ID:-}" != "debian" ] || { [ "${VERSION_ID:-}" != "12" ] && [ "${VERSION_ID:-}" != "13" ]; }; then
  echo "This installer supports Debian 12 and 13, not ${PRETTY_NAME:-this system}." >&2
  exit 1
fi
if [ "$(dpkg --print-architecture)" != "amd64" ]; then
  echo "This installer supports amd64 only." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  if [ "$(id -u)" != "0" ]; then
    echo "Python 3 is missing. Run the installer as root (sudo ./install.sh) so it can install it." >&2
    exit 1
  fi
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3
fi

exec python3 tools/install/strata_install.py "$@"
